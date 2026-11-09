"""Build a compact, real-data snapshot for the GitHub Pages app.

Run locally from the repository root. Raw bouts and biography caches never enter
the web build; only this explicitly selected analysis output is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

CATEGORIES = {
    'ME': ('Epee', 'Mens', "Men's epee"), 'WE': ('Epee', 'Womens', "Women's epee"),
    'MF': ('Foil', 'Mens', "Men's foil"), 'WF': ('Foil', 'Womens', "Women's foil"),
    'MS': ('Sabre', 'Mens', "Men's sabre"), 'WS': ('Sabre', 'Womens', "Women's sabre"),
}


def name_key(value):
    value = unicodedata.normalize('NFKD', str(value))
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return ' '.join(sorted(''.join(c.lower() if c.isalnum() else ' ' for c in value).split()))


def fit_scores(bouts, minimum):
    """Solve the same SpringRank objective as the research notebook, per league."""
    counts = pd.concat([bouts['Fencer A'], bouts['Fencer B']]).value_counts()
    eligible = set(counts[counts >= minimum].index)
    kept = bouts[bouts['Fencer A'].isin(eligible) & bouts['Fencer B'].isin(eligible)].copy()
    if kept.empty:
        raise ValueError(f'No bouts remain at minimum={minimum}')
    names = sorted(set(kept['Fencer A']) | set(kept['Fencer B']))
    ids = {name: i for i, name in enumerate(names)}
    a = kept['Fencer A'].map(ids).to_numpy()
    b = kept['Fencer B'].map(ids).to_numpy()
    wins = kept['Winner'].eq(kept['Fencer A']).to_numpy()
    adjacency = sparse.coo_matrix((np.ones(len(kept)), (np.where(wins, a, b), np.where(wins, b, a))),
                                  shape=(len(names), len(names))).tocsr()
    n_components, labels = connected_components(adjacency + adjacency.T, directed=False)
    largest = int(np.argmax(np.bincount(labels)))
    mask = labels == largest
    names = np.asarray(names)[mask]
    adjacency = adjacency[mask][:, mask]
    out_strength = np.asarray(adjacency.sum(axis=1)).ravel()
    in_strength = np.asarray(adjacency.sum(axis=0)).ravel()
    laplacian = sparse.diags(out_strength + in_strength) - adjacency - adjacency.T
    rhs = out_strength - in_strength
    scores = np.zeros(len(names))
    scores[1:] = spsolve(laplacian[1:, 1:], rhs[1:])
    scores -= scores.mean()
    residual = float(np.max(np.abs(laplacian @ scores - rhs)))
    if not np.isfinite(scores).all() or residual > 1e-7:
        raise ValueError(f'SpringRank did not solve accurately: residual={residual}')
    result = pd.DataFrame({'name': names, 'score': scores, 'retained_bouts': (out_strength + in_strength).astype(int)})
    result = result.sort_values(['score', 'name'], ascending=[False, True]).reset_index(drop=True)
    result['model_rank'] = result['score'].rank(ascending=False, method='min').astype(int)
    return result, {'input_bouts': len(bouts), 'retained_bouts': int(adjacency.sum()),
                    'fencers': len(names), 'components_before_filter': int(n_components),
                    'equation_residual': residual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--rankings', type=Path, help='Optional directory of archived FIE ranking CSVs')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / 'pages/public/analysis.json'
    rankings = args.rankings or root / 'rankings'
    inputs = [root / 'CoC dataset/Pools.csv', root / 'CoC dataset/DEs.csv']
    frames = [pd.read_csv(p, low_memory=False) for p in inputs]
    raw = pd.concat(frames, ignore_index=True)
    required = ['Fencer A', 'Fencer B', 'Winner', 'Weapon', 'Gender', 'Age Category', 'Bout Status']
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f'Missing bout columns: {missing}')
    valid = raw.dropna(subset=required).copy()
    valid = valid[valid['Bout Status'].eq('Fenced') & valid['Fencer A'].ne(valid['Fencer B'])]
    valid = valid[valid['Winner'].eq(valid['Fencer A']) | valid['Winner'].eq(valid['Fencer B'])]
    dates = pd.to_datetime(raw['Event Date'], format='mixed', errors='coerce')
    countries = pd.concat([
        valid[['Fencer A', 'Country A']].rename(columns={'Fencer A': 'name', 'Country A': 'country'}),
        valid[['Fencer B', 'Country B']].rename(columns={'Fencer B': 'name', 'Country B': 'country'}),
    ])
    # Country labels describe the most frequently recorded affiliation, not immutable nationality.
    country_lookup = countries.dropna().groupby('name')['country'].agg(lambda s: s.value_counts().index[0]).to_dict()
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_files = [{'path': str(p.relative_to(root)).replace('\\', '/'), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in inputs]
    fie = {}
    for code in CATEGORIES:
        path = rankings / f'{code}-detailed-ranking-2026.csv'
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if not {'Rank', 'Name'}.issubset(table.columns):
            raise ValueError(f'Invalid FIE ranking table: {path}')
        table['key'] = table['Name'].map(name_key)
        table['rank'] = pd.to_numeric(table['Rank'], errors='coerce')
        table = table[~table['key'].duplicated(keep=False) & table['rank'].gt(0)].dropna(subset=['rank'])
        fie[code] = table
        source_files.append({'path': f'rankings/{path.name}', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    athletes, identity_lookup, variants = [], {}, []
    for population in ['All recorded ages', 'Senior events']:
        population_bouts = valid if population == 'All recorded ages' else valid[valid['Age Category'].eq('Senior')]
        for minimum in [1, 5, 10]:
            for code, (weapon, gender, label) in CATEGORIES.items():
                group = population_bouts[population_bouts['Weapon'].eq(weapon) & population_bouts['Gender'].eq(gender)]
                scores, summary = fit_scores(group, minimum)
                scores['key'] = scores['name'].map(name_key)
                ambiguous = scores['key'].duplicated(keep=False)
                if code in fie:
                    scores = scores.merge(fie[code][['key', 'rank']], on='key', how='left', validate='many_to_one')
                    scores.loc[ambiguous.to_numpy(), 'rank'] = np.nan
                else:
                    scores['rank'] = np.nan
                rows = []
                for record in scores.to_dict('records'):
                    identity = (code, record['name'])
                    if identity not in identity_lookup:
                        identity_lookup[identity] = len(athletes)
                        athletes.append({'name': record['name'], 'country': country_lookup.get(record['name']), 'category': code})
                    rank = None if pd.isna(record['rank']) else int(record['rank'])
                    rows.append([identity_lookup[identity], round(record['score'], 10), int(record['model_rank']),
                                 record['retained_bouts'], rank])
                comparisons = {}
                for name, matched in [('all', scores.dropna(subset=['rank'])), ('top200', scores[scores['rank'].le(200)])]:
                    comparisons[name] = {'n': len(matched),
                        'spearman': float(stats.spearmanr(matched['score'], matched['rank']).statistic) if len(matched) >= 3 else None,
                        'kendall': float(stats.kendalltau(matched['score'], matched['rank']).statistic) if len(matched) >= 3 else None}
                variants.append({'population': population, 'minimum_bouts': minimum, 'category': code,
                                 **summary, 'comparison': comparisons, 'rows': rows})
    payload = {'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'source_commit': commit, 'model': 'SpringRank', 'source_files': source_files,
               'data_kind': 'actual_research_outputs',
               'category_labels': {c: v[2] for c, v in CATEGORIES.items()},
               'row_columns': ['athlete_id', 'score', 'model_rank', 'retained_bouts', 'fie_rank'],
               'audit': {'raw_bouts': len(raw), 'fenced_valid_bouts': len(valid), 'invalid_date_rows': int(dates.isna().sum()),
                         'first_parseable_date': str(dates.min().date()), 'last_parseable_date': str(dates.max().date())},
               'method': {'eligibility': 'Minimum recorded bouts is applied once before opponent and component filtering.',
                          'groups': 'Each weapon and gender is fitted independently. Mixed-gender events are excluded.',
                          'outcomes': 'Only Fenced rows with a winner equal to one of two distinct opponents are used.',
                          'component': 'Largest weakly connected component after eligibility filtering.',
                          'solver': 'Sparse direct solution of the SpringRank Laplacian objective, then mean-centered.',
                          'matching': 'Unique normalized names within each category; not license-verified identity resolution.',
                          'rankings': 'Archived 2025/26 FIE ranking CSVs; retrieval date not recorded in original files.',
                          'interpretation': 'Agreement with archived rankings is concurrent validation, not held-out prediction.'},
               'athletes': athletes, 'variants': variants}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temporary.replace(output)
    print(f'Exported {len(variants)} real analysis variants, {len(athletes)} athlete/category identities, {output.stat().st_size:,} bytes: {output}')


if __name__ == '__main__':
    raise SystemExit('Legacy schema-1 exporter retired. Open notebooks/research_analysis.py and run its visible analysis/export cells.')
