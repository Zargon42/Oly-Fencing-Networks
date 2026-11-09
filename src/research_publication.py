"""Format computed notebook results and write the reviewed export in one place.

No fitting or analysis runs on import. The notebook supplies every result.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse
from . import research, ranking_points
from .notebook_data import ZONE_CODES, ZONE_LABELS, RULE_SOURCE, json_safe

CATEGORY_LABELS = {'ME': "Men's epee", 'WE': "Women's epee", 'MF': "Men's foil",
                   'WF': "Women's foil", 'MS': "Men's sabre", 'WS': "Women's sabre"}


def fit_diagnostics(league, frame):
    ids = frame['athlete_index'].to_numpy(int)
    keep = np.isin(league.a, ids) & np.isin(league.b, ids)
    winners = np.where(league.winner_a, league.a, league.b)[keep]
    losers = np.where(league.winner_a, league.b, league.a)[keep]
    matrix = sparse.coo_matrix((np.ones(keep.sum()), (winners, losers)), shape=(len(league.names), len(league.names))).tocsr()
    outgoing = np.asarray(matrix.sum(axis=1)).ravel()
    incoming = np.asarray(matrix.sum(axis=0)).ravel()
    scores = np.zeros(len(league.names)); scores[ids] = frame['score']
    residual = (sparse.diags(outgoing + incoming) - matrix - matrix.T) @ scores - (outgoing - incoming)
    return {'input_bouts': len(league.a), 'equation_residual': float(np.abs(residual).max())}


def assemble_zones(frames, inference, adjusted, sensitivities, strength, prediction,
                   federations, audit, events, checks):
    joined = pd.concat(frames.values(), ignore_index=True)
    category_means = joined[joined['region'].isin(ZONE_CODES)].groupby(['category', 'region']).agg(
        n=('name', 'size'), mean=('D_original', 'mean'), with_zonal=('D_total_ties', 'mean'),
        without_zonal=('D_without_zonal', 'mean'), zonal_points=('zonal_points', 'mean')).reset_index()
    category_contrasts = []
    for code, frame in frames.items():
        row = {'category': code, 'n': len(frame)}
        for metric in ['D_total_ties', 'D_without_zonal', 'D_change']:
            row[metric] = research.regional_contrast(frame.assign(D=frame[metric]))
        category_contrasts.append(row)
    point_summary = joined.groupby('region').agg(n=('name', 'size'), mean_original_D=('D_original', 'mean'),
        mean_total_ties_D=('D_total_ties', 'mean'), mean_without_zonal_D=('D_without_zonal', 'mean'),
        mean_change=('D_change', 'mean'), mean_zonal_points=('zonal_points', 'mean'),
        mean_zonal_fraction=('zonal_fraction', 'mean'), mean_global_rank_change=('global_rank_change', 'mean')).reset_index()
    by_label = {row['analysis']: row for row in sensitivities}
    zonal = {'status': 'complete', 'regional_level': 'four_championship_zones',
             'zone_labels': [ZONE_LABELS[z] for z in ZONE_CODES], 'rule_source': RULE_SOURCE,
             'reconciliation': audit.to_dict('records'), 'events': events.to_dict('records'),
             'contrasts': inference['contrasts'], 'regional_intervals': inference['regional_intervals'],
             'regional_summary': point_summary.replace({'region': ZONE_LABELS}).to_dict('records'),
             'correlations': inference['correlations'], 'adjusted': adjusted,
             'category_regional_contrasts': category_contrasts,
             'top200_sensitivity': by_label['FIE top 200'],
             'both_zonal_points_and_bouts_removed': by_label['Exclude zonal bouts'],
             'bootstrap': {key: inference[key] for key in ['replicates', 'seed', 'joint_valid', 'signature']},
             'validation': checks,
             'tie_policy': 'Average point ties before and after within identical category-specific matched cohorts.',
             'interpretation': 'Mechanical points ablation with observed participation held fixed; not a causal estimate of athlete ability.'}
    zonal['bootstrap']['valid_fraction'] = inference['joint_valid'] / inference['replicates']
    return json_safe({'regional_level': 'four_championship_zones', 'zone_labels': zonal['zone_labels'],
        'zone_code_labels': ZONE_LABELS,
        'hypothesis': 'The zonal slot benefits FIE placement relative to SpringRank more for Africa/Americas than Asia-Oceania/Europe, with a complementary test at comparable non-zonal strength.',
        'category_means': category_means.replace({'region': ZONE_LABELS}).to_dict('records'),
        'unknown_affiliation_rows': int(joined['region'].eq('Unknown').sum()),
        'original_rank_contrast': inference['contrasts']['D_original'], 'adjusted': adjusted,
        'sensitivities': sensitivities, 'zonal_points': zonal, 'strength_conditioned': strength,
        'prediction': prediction, 'federation_classification': federations,
        'classification_sources': ['https://www.ausfencing.org/team-announcement/2025-senior-asian-championships/',
                                   'https://fie.org/tournaments/2024/1067/event/1067/pools']})


def build_snapshot(root, raw, bouts, variant_fits, variant_leagues, noise_summary,
                   noise_bootstrap, beta, rho, zones, confederations, biography_coverage):
    """Construct all 36 app selections from notebook fits, not a previous export."""
    identities = {}
    for (_, _, code), (frame, _) in variant_fits.items():
        for row in frame.to_dict('records'):
            identities[(code, row['name'])] = row['country']
    keys = sorted(identities)
    ids = {key: index for index, key in enumerate(keys)}
    athletes = [{'category': code, 'name': name, 'country': identities[(code, name)]} for code, name in keys]
    variants, summaries = [], []
    for (population, minimum, code), (frame, summary) in variant_fits.items():
        comparison = {}
        for label, matched in [('all', frame.dropna(subset=['fie_rank'])),
                               ('top200', frame[frame['fie_rank'].le(200)])]:
            comparison[label] = {'n': len(matched), 'spearman': research.spearman(matched['score'], matched['fie_rank'])}
        rows = [[ids[(code, r['name'])], round(float(r['score']), 10), int(r['model_rank']),
                 int(r['retained_bouts']), int(r['fie_rank']) if pd.notna(r['fie_rank']) else None]
                for r in frame.to_dict('records')]
        variants.append({'population': population, 'minimum_bouts': minimum, 'category': code,
                         'fencers': len(frame), 'retained_bouts': summary['bouts'], 'rows': rows,
                         'beta': summary['beta'], 'beta_status': summary['beta_status'],
                         'depth': summary['depth'], 'levels': summary['levels'], 'comparison': comparison,
                         **fit_diagnostics(variant_leagues[population][code], frame)})
        summaries.append({'population': population, 'minimum': minimum, **summary})
    source_paths = [Path('CoC dataset/Pools.csv'), Path('CoC dataset/DEs.csv')]
    source_paths += [Path(f'rankings/{code}-detailed-ranking-2026.csv') for code in research.CATEGORIES]
    sources = [{'path': str(path).replace('\\', '/'), 'sha256': hashlib.sha256((Path(root) / path).read_bytes()).hexdigest()}
               for path in source_paths]
    dates = bouts['Date']
    return json_safe({'schema_version': 3, 'data_kind': 'actual_research_outputs', 'model': 'SpringRank',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'category_labels': CATEGORY_LABELS,
        'row_columns': ['athlete_id', 'score', 'model_rank', 'retained_bouts', 'fie_rank'],
        'source_files': sources, 'athletes': athletes, 'variants': variants,
        'audit': {'raw_bouts': len(raw), 'fenced_valid_bouts': len(bouts), 'invalid_date_rows': int(dates.isna().sum()),
                  'first_parseable_date': str(dates.min().date()), 'last_parseable_date': str(dates.max().date())},
        'research': {'primary': noise_summary.to_dict('records'), 'beta': beta, 'rho': rho,
                     'bootstrap': {'replicates': len(noise_bootstrap['beta']), 'seed': noise_bootstrap['seed'],
                                   'signature': noise_bootstrap['signature'],
                                   'cohort_min': noise_bootstrap['sizes'].min(axis=0),
                                   'cohort_max': noise_bootstrap['sizes'].max(axis=0),
                                   'failed_fit_records': sum(len(x) for x in noise_bootstrap['failures'])},
                     'sensitivity_summary': summaries, 'region_intervals': confederations,
                     'four_zone_primary': zones, 'biography_coverage': biography_coverage,
                     'region_source': 'https://fie.org/federations/FCA',
                     'notebook': 'https://github.com/Zargon42/Oly-Fencing-Networks/blob/main/notebooks/research_analysis.py',
                     'caveat': 'Conditional on observed athletes and event coverage. SpringRank is not true ability; beta is not intrinsic luck. Four championship zones are primary; five confederations are a secondary description.'}})


def markdown_table(rows, columns=None):
    frame = pd.DataFrame(rows)
    if columns is not None: frame = frame[columns]
    if frame.empty: return 'No results available.'
    def text(value):
        if isinstance(value, (float, np.floating)): return f'{value:.4g}' if np.isfinite(value) else 'NA'
        return str(value).replace('|', '/').replace('\n', ' ')
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    rule = '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'
    body = ['| ' + ' | '.join(text(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, rule, *body])


def research_report(snapshot):
    """A readable report with tables, not a dump of the snapshot JSON."""
    r = snapshot['research']; z = r['four_zone_primary']; p = z['zonal_points']
    contrasts = p['contrasts']; strength = z['strength_conditioned']; prediction = z['prediction']
    beta_test, rho_test = r['beta'].get('omnibus'), r['rho'].get('omnibus')
    beta_p = f"{beta_test['p']:.3f}" if beta_test else 'suppressed (insufficient valid replicates)'
    rho_p = f"{rho_test['p']:.3f}" if rho_test else 'suppressed (insufficient valid replicates)'
    before, after, change = [contrasts[k] for k in ['D_total_ties', 'D_without_zonal', 'D_change']]
    reward = next(row for row in strength['models'] if row['outcome'] == 'zonal_points')
    basic = z['adjusted']['D_without_zonal']['basic']['contrast']
    parts = [
        '# Fencing networks: noise, ranking discrepancies, and zonal rewards',
        f"**Draft for manual review.** Generated {snapshot['generated_at_utc'][:10]} by the executable research notebook.",
        '## Research questions and population\nDo six weapon/gender categories differ in model-estimated noise or FIE agreement? Are ranking discrepancies associated with four championship zones, and do zonal rewards differ at comparable non-zonal SpringRank strength? Can biographies predict snapshot disagreement or later bouts? Primary population: senior events, at least five recorded bouts. Each category is fitted separately.',
        '## 1. Noise across the six categories\nThe corrected conditional likelihood counts every unordered opponent pair once. Scores use a common SpringRank convention; independently standardizing scores would change beta. Win probability is logistic(2 beta (score A - score B)). Higher beta is a sharper relationship, not proof of less intrinsic luck.',
        markdown_table(r['primary'], ['category', 'beta', 'legacy_beta', 'depth', 'levels', 'rho', 'matched', 'athletes', 'bouts']),
        f"Joint tournament-block bootstrap: {r['bootstrap']['replicates']:,} draws, seed {r['bootstrap']['seed']}; filtering, scores and matching are refitted. Beta omnibus p={beta_p}; correlation omnibus p={rho_p}. Failure to reject does not establish equivalence. Correlations use score versus official rank number, so stronger agreement is more negative. Formal inference is suppressed below 90% joint validity.",
        '### Beta intervals\n' + markdown_table(r['beta']['intervals']),
        '### FIE correlation intervals\n' + markdown_table(r['rho']['intervals']),
        '### Pairwise beta contrasts (separate Holm family)\n' + markdown_table(r['beta']['pairs']),
        '### Pairwise correlation contrasts (separate Holm family)\n' + markdown_table(r['rho']['pairs']),
        '## 2. Ranking discrepancy and the four-zone hypothesis\nD=(SpringRank rank-FIE rank)/(n-1), with both orders reranked in the common category-specific cohort and average ties. Positive D means better FIE placement. The primary equal-zone contrast is (Africa+Americas-Asia-Oceania-Europe)/2. Oceania retains separate confederation membership but shares the Asian championship pool. Unresolved affiliations are explicit and excluded from zone regressions. This is an exploratory hypothesis, not a claim of intrinsic regional strength.',
        '## 3. Reconciled zonal-points counterfactual\nThe five best ordinary-event scores plus separate world/Olympic and zonal slots reproduce every supplied total. Removing the zonal slot does not promote a sixth ordinary result. Average point ties are used both before and after, with participation fixed. Exact official placement tie-breaks are not reconstructed.',
        markdown_table(p['reconciliation'], ['category', 'athletes', 'headers_repaired', 'reconciled', 'unreconciled', 'max_absolute_error']),
        f"The tie-controlled contrast changes from {before['estimate']:.4f} to {after['estimate']:.4f}. Paired change {change['estimate']:.4f}, 95% tournament interval [{change['low']:.4f}, {change['high']:.4f}]. Relative reduction {contrasts['fraction_reduction']['estimate']:.1%}. This is a mechanical change, not the causal proportion explained. The category/age/hand-adjusted residual contrast is {basic['estimate']:.4f} [{basic['low']:.4f}, {basic['high']:.4f}].",
        markdown_table([{'metric': k, **v} for k, v in contrasts.items()]),
        '### Adjusted equal-zone contrasts\n' + markdown_table([{'outcome': target, 'adjustment': kind, **model['contrast']} for target, models in z['adjusted'].items() for kind, model in models.items()]),
        '### Four-zone sensitivities\n' + markdown_table([{'analysis': row['analysis'], 'n': row['n'], **row['contrasts']} for row in z['sensitivities']]),
        '## 4. Rewards at comparable estimated strength\nStrength is a within-category percentile from SpringRank fitted without zonal bouts. Participation evidence is a recorded zonal score or observed zonal bout; absence of evidence does not establish nonparticipation. The intersection of zone-specific 5th-to-95th percentile ranges limits extrapolation. Models use category-specific cubic strength curves, category, age and hand; uncertainty is country-clustered, with Holm correction for two exploratory outcomes.',
        f"Among {reward['n']:,} supported athletes, the equal-zone reward contrast is {reward['estimate']:.2f} points [{reward['low']:.2f}, {reward['high']:.2f}]. Strength uncertainty is not fully propagated in this conditional model. The association cannot establish ability, unfairness or a causal effect of entering zonals.",
        markdown_table(strength['models'], ['outcome', 'n', 'countries', 'estimate', 'low', 'high', 'p_holm']),
        '### Common support and exclusions\n' + markdown_table(strength['support_audit'], ['category', 'eligible', 'retained', 'low', 'high']),
        '### Excluding sparse retained category/zone cells\n' + markdown_table(strength['sensitivity_models'], ['outcome', 'n', 'estimate', 'low', 'high', 'p']),
        '## 5. Prediction tasks\nSnapshot discrepancy prediction uses country-grouped outer validation and grouped inner tuning; ambiguous cross-country names are conservatively excluded. This is not future-ranking forecasting. Bout prediction uses rolling whole-event holdouts, training-only scores, beta and imputation. Invalid dates are excluded. Clubs and coaches are not treated as historically known.',
        '### Discrepancy prediction\n' + markdown_table(prediction['discrepancy']['metrics']),
        '### Later bouts: all categories\n' + markdown_table([row for row in prediction['bout']['metrics'] if row['category'] == 'All']),
        f"Bout coverage {prediction['bout']['coverage']:.1%}. Paired event-bootstrap Brier improvement: {prediction['bout']['brier_improvement']}. The logistic model also recalibrates the score coefficient; differences cannot be attributed exclusively to biographies.",
        '## 6. Interpretation and limits\nThe recorded data support a zonal-reward mechanism for the observed ranking discrepancy. SpringRank is an imperfect comparison, not true ability. Entry selection, federation quotas, missing biographies, name matching and event coverage limit interpretation. Tournament intervals condition on the fixed archived FIE snapshot; independent seasons and historical rankings are needed for replication and forecasting. Five-confederation descriptions are secondary. Height and club coverage are sparse; primary biography models use age and hand.',
        '## Fantasy matchups\nWithin one league: logistic(2 beta (sA-sB)). Across leagues: z=2 beta (s-mean), then logistic(zA-zB). Aligning means is an entertainment convention; disconnected leagues have no empirically established common ability scale.',
        '## Reproducibility and sources\nRun notebooks/research_analysis.py from a clean marimo session with the local inputs described in DATA.md. The last section is the only export writer. Raw biographies and this draft stay outside Pages. Full machine-readable results are separate from this report.\n\nSpringRank: https://pmc.ncbi.nlm.nih.gov/articles/PMC6054508/\nFIE counting rules: ' + RULE_SOURCE + '\nFederations: https://fie.org/federations/FCA\nBout source: Andrew Fischl (CyrusOfChaos), as credited in DATA.md.'
    ]
    return '\n\n'.join(parts) + '\n'


def write_exports(root, snapshot, report, tables, events):
    root = Path(root)
    destinations = {'snapshot': root / 'pages/public/analysis.json',
                    'report': root / 'Analysis/research_report.md'}
    for key, value in [('snapshot', json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'), allow_nan=False)),
                       ('report', report)]:
        path = destinations[key]; path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(value, encoding='utf-8'); temporary.replace(path)
    clean = root / 'data/research/clean_rankings'; clean.mkdir(parents=True, exist_ok=True)
    for code, table in tables.items(): table.to_csv(clean / f'{code}-reconciled.csv', index=False)
    events.to_csv(clean / 'event_dictionary.csv', index=False)
    # Keep older local links usable without keeping an obsolete appended report.
    (root / 'Analysis/research_report.txt').write_text(report, encoding='utf-8')
    return destinations
