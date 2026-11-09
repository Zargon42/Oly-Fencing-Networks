"""Input cleaning and small transformations used by the research notebook.

Nothing runs on import. Statistical fits and their run order belong to notebook
cells; this module keeps parsing and presentation plumbing out of that narrative.
"""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import re

import numpy as np
import pandas as pd
import requests
from . import research, ranking_points

ZONE_CODES = ['Africa', 'Americas', 'Asia', 'Europe']
ZONE_LABELS = {'Africa': 'Africa', 'Americas': 'Americas', 'Asia': 'Asia-Oceania',
               'Europe': 'Europe', 'Unknown': 'Unknown'}
RULE_SOURCE = 'https://static.fie.org/uploads/38/190670-Organisation%20rules%20ang.pdf'


def federation_lookup(cache_path):
    """Cache only federation names/codes and confederations, never contacts."""
    path = Path(cache_path)
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    url = 'https://fie.org/federations/FCA'
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=40)
    response.raise_for_status()
    source = {'source_url': url, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
              'federations': research.official_regions(response.text)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source, indent=2), encoding='utf-8')
    return source


def reconcile_rankings(folder, season=2026):
    """Repair rotated headers and reconcile the best-five plus separate slots."""
    tables, events, audit = {}, [], []
    for code in research.CATEGORIES:
        path = Path(folder) / f'{code}-detailed-ranking-{season}.csv'
        table = pd.read_csv(path)
        rename, ordinary, zonal, world = {}, [], [], []
        for header in table.columns:
            if header in ['Rank', 'Name', 'Nat.', 'Points', 'Total points']:
                continue
            text = ' '.join(str(header).split())
            reversed_header = text.startswith(')')
            if reversed_header:
                text = ' '.join(part[::-1] for part in reversed(text.split()))
            match = re.fullmatch(r'(\d{2})/(\d{2})\s+(.+?)\s*\((SA|A|GP|CHZ|CHM|JO)\)', text)
            if not match:
                raise ValueError(f'Unrecognized {code} tournament header: {header!r}')
            day, month, location, kind = match.groups()
            year = season - 1 if int(month) >= 9 else season
            date = pd.Timestamp(year=year, month=int(month), day=int(day))
            clean = f'{date.date()} {location} ({kind})'
            assert clean not in rename.values(), 'Duplicate cleaned tournament header'
            rename[header] = clean
            if kind in ['SA', 'A', 'GP']: ordinary.append(clean)
            elif kind == 'CHZ': zonal.append(clean)
            else: world.append(clean)
            events.append({'category': code, 'original_header': header, 'clean_header': clean,
                           'date': str(date.date()), 'location': location, 'event_type': kind,
                           'reversed_header_fixed': reversed_header})
        table = table.rename(columns={**rename, 'Rank': 'official_rank', 'Name': 'name',
                                     'Nat.': 'country', 'Points': 'official_total', 'Total points': 'official_total'})
        assert len(zonal) == 4 and len(world) == 1
        points = table[list(rename.values())].apply(pd.to_numeric, errors='raise')
        assert points.fillna(0).ge(0).all().all()
        assert points[zonal].fillna(0).gt(0).sum(axis=1).le(1).all()
        table['ordinary_best_five'] = np.sort(points[ordinary].fillna(0).to_numpy(), axis=1)[:, -5:].sum(axis=1)
        table['ordinary_all_points'] = points[ordinary].sum(axis=1)
        table['discarded_ordinary_points'] = table['ordinary_all_points'] - table['ordinary_best_five']
        table['zonal_points'] = points[zonal].sum(axis=1)
        table['world_points'] = points[world].sum(axis=1)
        table['reconstructed_total'] = table['ordinary_best_five'] + table['zonal_points'] + table['world_points']
        table['reconciliation_error'] = table['reconstructed_total'] - table['official_total']
        table['reconciled'] = np.isclose(table['reconciliation_error'], 0, atol=1e-8)
        table['points_without_zonal'] = table['ordinary_best_five'] + table['world_points']
        table['rank_from_total_average_ties'] = table['official_total'].rank(ascending=False)
        table['rank_without_zonal_average_ties'] = table['points_without_zonal'].rank(ascending=False)
        table['key'] = table['name'].map(research.normalized_name)
        tables[code] = table
        audit.append({'category': code, 'athletes': len(table), 'events': len(rename),
                      'headers_repaired': sum(e['reversed_header_fixed'] for e in events if e['category'] == code),
                      'reconciled': int(table['reconciled'].sum()), 'unreconciled': int((~table['reconciled']).sum()),
                      'max_absolute_error': float(table['reconciliation_error'].abs().max()),
                      'athletes_with_zonal_points': int(table['zonal_points'].gt(0).sum()),
                      'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return tables, pd.DataFrame(events), pd.DataFrame(audit)


def championship_leagues(leagues):
    return {code: replace(league, region=np.where(league.region == 'Oceania', 'Asia', league.region))
            for code, league in leagues.items()}


def championship_fits(fits):
    result = {}
    for code, (frame, summary) in fits.items():
        frame = frame.copy()
        frame['confederation_region'] = frame['region']
        frame['region'] = frame['region'].replace({'Oceania': 'Asia'})
        result[code] = frame, summary
    return result


def championship_federations(federations):
    return {code: {**row, 'confederation_region': row['region'],
                   'championship_zone': 'Asia' if row['region'] == 'Oceania' else row['region'],
                   'region': 'Asia' if row['region'] == 'Oceania' else row['region']}
            for code, row in federations.items()}


def points_frames(fits, tables, elite=False):
    result = {}
    for code, (frame, _) in fits.items():
        if elite:
            frame = frame.query('fie_rank <= 200').copy()
            frame['D'] = research.discrepancy(frame['score'], frame['fie_rank'])
        result[code] = ranking_points.attach_points(frame, tables[code])
    return result


def adjusted_contrasts(features, point_data):
    results = {}
    for target in ['D_original', 'D_total_ties', 'D_without_zonal', 'D_change']:
        data = features.drop(columns=['D']).merge(point_data[['category', 'name', target]],
                                                 on=['category', 'name'], validate='one_to_one')
        data['D'] = data[target]
        results[target] = {kind: research.clustered_adjustment(data, participation)
                           for kind, participation in [('basic', False), ('participation', True)]}
    return results


def points_inference(frames, bootstrap):
    """Summarize an already-computed paired bootstrap; never fit networks here."""
    point, correlations = ranking_points.summarize(list(frames.values()))
    valid = [row for row in bootstrap['results'] if row.get('failure') is None]
    requested = bootstrap['replicates']
    contrasts = {key: ranking_points.interval(value, [r['contrasts'][key] for r in valid], requested)
                 for key, value in point.items()}
    correlation_intervals = []
    for index, row in enumerate(correlations):
        result = {'category': row['category'], 'n': row['n']}
        for metric in ['rho_total_ties', 'rho_without_zonal', 'rho_change']:
            result[metric] = ranking_points.interval(row[metric], [r['correlations'][index][metric] for r in valid], requested)
        correlation_intervals.append(result)
    pvalues = [r['rho_change']['p'] for r in correlation_intervals]
    if all(p is not None for p in pvalues):
        for row, p in zip(correlation_intervals, research.holm(pvalues)):
            row['rho_change']['p_holm'] = float(p)
    joined = pd.concat(frames.values(), ignore_index=True)
    regional = []
    for i, region in enumerate(ZONE_CODES):
        subset = joined[joined['region'].eq(region)]
        for j, metric in enumerate(['D_total_ties', 'D_without_zonal', 'D_change']):
            regional.append({'region': ZONE_LABELS[region], 'metric': metric, 'n': len(subset),
                             **ranking_points.interval(float(subset[metric].mean()), [r['regional'][i][j] for r in valid], requested)})
    return {'contrasts': contrasts, 'correlations': correlation_intervals, 'regional_intervals': regional,
            'joint_valid': len(valid), 'replicates': requested, 'seed': bootstrap['seed'],
            'signature': bootstrap['signature']}


def confederation_intervals(fits, bootstrap):
    rows = []
    for i, (category, (frame, _)) in enumerate(fits.items()):
        for j, region in enumerate(research.REGIONS):
            subset = frame[frame['region'].eq(region)].dropna(subset=['D'])
            values = bootstrap['regional_means'][:, i, j]
            values = values[np.isfinite(values)]
            rows.append({'category': category, 'region': region, 'n': len(subset),
                         'mean': float(subset['D'].mean()), 'low': float(np.quantile(values, .025)),
                         'high': float(np.quantile(values, .975)), 'valid_replicates': len(values)})
    return rows


def category_zone_means(frames):
    data = pd.concat(frames.values(), ignore_index=True)
    data = data[data['region'].isin(ZONE_CODES)]
    return data.groupby(['category', 'region']).agg(n=('name', 'size'), mean=('D_original', 'mean'),
        with_zonal=('D_total_ties', 'mean'), without_zonal=('D_without_zonal', 'mean'),
        zonal_points=('zonal_points', 'mean')).reset_index().replace({'region': ZONE_LABELS})


def svg(figure):
    buffer = io.StringIO()
    figure.savefig(buffer, format='svg', bbox_inches='tight')
    return buffer.getvalue().replace('<svg ', '<svg style="max-width:100%;height:auto" ', 1)


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_): return bool(value)
    return value
