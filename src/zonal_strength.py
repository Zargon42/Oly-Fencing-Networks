"""Conditional reward models called by the zonal-strength notebook section."""
import numpy as np
import pandas as pd
from scipy import stats
from . import research
from .notebook_data import ZONE_CODES


def prepare_strength_cohort(point_data, nonzonal_fits, features, tables, lower=.05, upper=.95):
    """Build participation evidence and restrict each category to common support."""
    abilities = []
    for code, (frame, _) in nonzonal_fits.items():
        frame = frame[['category', 'name', 'score']].copy()
        frame['ability_percentile'] = frame['score'].rank(method='average', pct=True)
        abilities.append(frame.rename(columns={'score': 'nonzonal_score'}))
    data = point_data.merge(pd.concat(abilities), on=['category', 'name'], how='left', validate='one_to_one')
    data = data.merge(features[['category', 'name', 'age', 'hand', 'zonal_share']], on=['category', 'name'], validate='one_to_one')
    records = []
    for code, table in tables.items():
        columns = [column for column in table if column.endswith('(CHZ)')]
        record = table.loc[~table['key'].duplicated(keep=False), ['key']].copy()
        record['recorded_zonal_score'] = table.loc[record.index, columns].notna().any(axis=1)
        record['category'] = code
        records.append(record)
    data = data.merge(pd.concat(records), on=['category', 'key'], validate='one_to_one')
    data['participation_evidence'] = data['recorded_zonal_score'] | data['zonal_share'].gt(0)
    data['rank_benefit'] = data['D_total_ties'] - data['D_without_zonal']
    coverage = data.groupby('region').agg(matched=('name', 'size'), participation_evidence=('participation_evidence', 'sum'),
                                         nonzonal_strength_available=('nonzonal_score', 'count')).reset_index()
    eligible = data[data['region'].isin(ZONE_CODES) & data['participation_evidence'] & data['ability_percentile'].notna()]
    descriptions = eligible.groupby(['category', 'region']).agg(n=('name', 'size'), mean_strength_percentile=('ability_percentile', 'mean'),
        mean_zonal_points=('zonal_points', 'mean'), mean_rank_benefit=('rank_benefit', 'mean')).reset_index()
    retained, audit = [], []
    for code, frame in eligible.groupby('category', sort=False):
        ranges = frame.groupby('region')['ability_percentile'].agg(['count', lambda x: x.quantile(lower), lambda x: x.quantile(upper)]).reindex(ZONE_CODES)
        ranges.columns = ['n', 'low', 'high']
        valid = bool(ranges['n'].ge(5).all())
        low, high = float(ranges['low'].max()), float(ranges['high'].min())
        subset = frame[frame['ability_percentile'].between(low, high)] if valid and low < high else frame.iloc[:0]
        audit.append({'category': code, 'eligible': len(frame), 'retained': len(subset), 'low': low, 'high': high,
                      'all_four_zones_have_five': valid, 'retained_by_zone': subset['region'].value_counts().to_dict()})
        retained.append(subset)
    supported = pd.concat(retained, ignore_index=True)
    assert set(ZONE_CODES).issubset(supported['region'].unique())
    return supported, coverage, pd.DataFrame(audit), descriptions


def conditioned_model(data, target, degree=3):
    """OLS with category-specific strength curves and country-clustered covariance."""
    data = data.copy()
    data['region'] = pd.Categorical(data['region'], categories=['Europe', 'Africa', 'Americas', 'Asia'])
    data['hand'] = pd.Categorical(data['hand'].fillna('Unknown'), categories=['Right', 'Left', 'Ambidextrous', 'Unknown'])
    design = pd.get_dummies(data[['region', 'category', 'hand']], drop_first=True, dtype=float)
    design['age_missing'] = data['age'].isna().astype(float)
    design['age'] = (data['age'].fillna(data['age'].median()) - 25) / 10
    for code in research.CATEGORIES:
        for power in range(1, degree + 1):
            design[f'{code}_strength_power_{power}'] = data['category'].eq(code).astype(float) * (data['ability_percentile'] - .5) ** power
    design = design.loc[:, design.nunique() > 1]
    design.insert(0, 'Intercept', 1.)
    matrix, y = design.to_numpy(float), data[target].to_numpy(float)
    inverse = np.linalg.pinv(matrix.T @ matrix)
    coefficients = inverse @ matrix.T @ y
    residual = y - matrix @ coefficients
    countries = data['country'].to_numpy()
    clusters, n, rank = len(np.unique(countries)), len(y), np.linalg.matrix_rank(matrix)
    assert clusters > 10 and n > rank
    meat = np.zeros_like(inverse)
    for country in np.unique(countries):
        mask = countries == country
        score = matrix[mask].T @ residual[mask]
        meat += np.outer(score, score)
    covariance = inverse @ meat @ inverse * (clusters / (clusters - 1)) * ((n - 1) / (n - rank))
    contrast = np.zeros(len(design.columns))
    for name, weight in [('region_Africa', .5), ('region_Americas', .5), ('region_Asia', -.5)]:
        contrast[design.columns.get_loc(name)] = weight
    estimate = float(contrast @ coefficients)
    se = float(np.sqrt(max(contrast @ covariance @ contrast, 0)))
    critical = stats.t.ppf(.975, clusters - 1)
    return {'outcome': target, 'n': n, 'countries': clusters, 'estimate': estimate,
            'low': float(estimate - critical * se), 'high': float(estimate + critical * se),
            'p': float(2 * stats.t.sf(abs(estimate / se), clusters - 1)) if se > 0 else None,
            'conditioning': 'Category-specific cubic in non-zonal SpringRank strength percentile, category, age and hand; country-clustered conditional uncertainty.'}


def dense_support(data, minimum=5):
    counts = data.groupby(['category', 'region']).size().unstack(fill_value=0).reindex(columns=ZONE_CODES, fill_value=0)
    categories = counts.index[counts.min(axis=1) >= minimum].tolist()
    return data[data['category'].isin(categories)], categories
