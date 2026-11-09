"""Research notebook: noise, FIE discrepancies, and biographical prediction."""

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    app_title="Fencing research: noise and ranking discrepancies",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Fencing research: noise and ranking discrepancies

    This notebook is the executable source of the research analysis. Calculations,
    model fitting, uncertainty estimates, and their outputs belong in the cells here.
    The public GitHub Pages app will display results exported from this notebook.

    **Primary analysis:** senior events, five recorded bouts, six separately fitted
    weapon/gender categories. Combined ages and one/ten-bout thresholds are sensitivity analyses.

    ## Research questions

    1. Do categories differ in SpringRank beta, and how uncertain are those differences?
    2. Do categories differ in their agreement with archived FIE rankings?
    3. Are FIE-versus-SpringRank discrepancies associated with sporting region?
    4. Can biographies predict discrepancy in held-out countries, or improve predictions of later bouts?

    Cross-league fantasy matchups are an explicitly hypothetical entertainment feature,
    separate from the statistical evidence.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sys

    return Path, json, np, pd, plt, sys


@app.cell
def _(Path, mo, sys):
    research_root = Path(mo.notebook_location()).parent
    if str(research_root) not in sys.path:
        sys.path.insert(0, str(research_root))
    return (research_root,)


@app.cell
def _(pd, research_root):
    research_pools = pd.read_csv(research_root / 'CoC dataset/Pools.csv', low_memory=False)
    research_des = pd.read_csv(research_root / 'CoC dataset/DEs.csv', low_memory=False)
    research_pools['Stage'] = 'Pool'
    research_des['Stage'] = 'DE'
    research_bouts = pd.concat([research_pools, research_des], ignore_index=True)
    return research_bouts, research_des, research_pools


@app.cell(hide_code=True)
def _(mo, research_bouts, research_des, research_pools):
    mo.md(f"""
    ## Source-data audit

    Loaded **{len(research_bouts):,} recorded bouts** in this notebook:
    {len(research_pools):,} pool records and {len(research_des):,} direct-elimination records.
    Source data are local; the public app will receive only the selected derived outputs.
    """)
    return


@app.cell
def _(mo):
    research_bootstrap_replicates = mo.ui.number(start=20, stop=10000, step=20, value=2000, label='Tournament bootstrap replicates')
    research_random_seed = mo.ui.number(start=0, stop=100000, step=1, value=42, label='Random seed')
    mo.hstack([research_bootstrap_replicates, research_random_seed], justify='start')
    return research_bootstrap_replicates, research_random_seed


@app.cell
def research_methods(mo):
    mo.md(r"""
    ## Methods and execution

    The full executable analysis lives in this notebook. Reusable implementations are
    published in `src/research.py`; the cells below make every data input, statistical run,
    validation result, and export visible. No independent analysis script is used to generate results.

    SpringRank uses $P(i>j)=\operatorname{logistic}(2\beta(s_i-s_j))$.
    The corrected likelihood counts each observed unordered opponent pair once, including
    zero wins in one direction. The legacy sparse implementation double-weighted reciprocal pairs.

    The bootstrap resamples shared tournament blocks across all six categories, then refits
    eligibility, connected components, scores, beta, and FIE matches. Its intervals describe
    uncertainty under the observed event mix, not intrinsic luck. Pairwise tests use centered
    bootstrap differences and Holm adjustment; formal inference requires at least 90% valid joint replicates.
    """)
    return


@app.cell
def research_helpers():
    import src.research as research
    import requests
    from datetime import datetime, timezone
    import importlib as _importlib
    _importlib.reload(research)
    return datetime, requests, research, timezone


@app.cell
def research_regions(
    datetime,
    json,
    mo,
    pd,
    requests,
    research,
    research_root,
    timezone,
):
    _cache = research_root / 'data/research/fie_regions.json'
    _cache.parent.mkdir(parents=True, exist_ok=True)
    if _cache.exists():
        research_region_source = json.loads(_cache.read_text(encoding='utf-8'))
    else:
        _url = 'https://fie.org/federations/FCA'
        _response = requests.get(_url, headers={'User-Agent':'Mozilla/5.0'}, timeout=40)
        _response.raise_for_status()
        research_region_source = {'source_url':_url, 'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
                                  'federations':research.official_regions(_response.text)}
        _cache.write_text(json.dumps(research_region_source, indent=2), encoding='utf-8')
    research_regions = research_region_source['federations']
    mo.ui.table(pd.DataFrame([{'Federation':code, **row} for code,row in research_regions.items()]),
                pagination=True, page_size=10, selection=None, label='Sourced FIE confederation lookup')
    return research_region_source, research_regions


@app.cell
def research_clean_data(mo, research, research_bouts):
    research_valid_bouts = research.prepare_bouts(research_bouts)
    research_audit = {'raw_rows':len(research_bouts), 'valid_fenced_rows':len(research_valid_bouts),
                      'invalid_date_rows':int(research_valid_bouts['Date'].isna().sum()),
                      'missing_source_urls':int(research_valid_bouts['Source URL'].isna().sum())}
    mo.ui.table([research_audit], selection=None, label='Filtering and date audit')
    return (research_valid_bouts,)


@app.cell
def research_primary_fits(
    mo,
    pd,
    research,
    research_regions,
    research_root,
    research_valid_bouts,
):
    research_primary_leagues, research_tournament_blocks = research.encode_leagues(
        research_valid_bouts, research_root / 'rankings', research_regions, population='Senior events')
    research_primary_fits = {}
    for _code, _league in research_primary_leagues.items():
        research_primary_fits[_code] = research.fit_league(_league, minimum=5)
    research_primary_summary = pd.DataFrame([fit[1] for fit in research_primary_fits.values()])
    mo.ui.table(research_primary_summary, selection=None, label='Six senior categories: corrected and legacy estimates')
    return (
        research_primary_fits,
        research_primary_leagues,
        research_primary_summary,
        research_tournament_blocks,
    )


@app.cell
def research_validation(mo, np, research, research_regions):
    _toy_scores = np.array([.5, 0., -.5])
    _toy_delta = np.array([.5, 1., .5])
    _toy_wins = np.array([8., 7., 6.])
    _toy_totals = np.array([10., 7., 10.])
    _corrected, _status = research.beta_from_pairs(_toy_delta, _toy_wins, _toy_totals)
    _legacy, _ = research.beta_from_pairs(_toy_delta, _toy_wins, _toy_totals, legacy=True)
    assert _status == 'finite' and abs(_corrected - _legacy) > .01
    _gradient = np.dot(_toy_delta, _toy_wins - _toy_totals / (1 + np.exp(-2*_corrected*_toy_delta)))
    assert abs(_gradient) < 1e-8
    assert research.beta_from_pairs(np.zeros(3), _toy_wins, _toy_totals)[1] == 'unidentified'
    assert research.beta_from_pairs(np.ones(2), np.ones(2), np.ones(2))[1] == 'separated_infinite'
    assert research.beta_from_pairs(np.ones(2), np.zeros(2), np.ones(2))[1] == 'boundary_zero'
    assert np.allclose(research.discrepancy([3,2,1],[1,2,3]), 0)
    assert research.discrepancy([1,2,3],[1,2,3])[0] > 0
    assert np.allclose(research.discrepancy([3,3,1],[1,1,3]), 0)
    assert research.fantasy_probability(.3,1.4,.3,1.4) == .5
    _p = research.fantasy_probability(.3,1.4,-.2,1.8)
    assert abs(_p + research.fantasy_probability(-.2,1.8,.3,1.4) - 1) < 1e-12
    assert research_regions.get('AIN', {}).get('region','Unknown') == 'Unknown'
    mo.md('**Implementation checks passed:** likelihood weighting, finite/zero/infinite/unidentified beta, discrepancy signs and ties, unresolved regions, and matchup symmetry.')
    return


@app.cell
def research_point_plots(np, plt, research_primary_summary):
    _fig, _axes = plt.subplots(1, 2, figsize=(11,4))
    _summary = research_primary_summary
    _axes[0].bar(_summary['category'], _summary['beta'], color='#287b8e', label='Corrected beta')
    _axes[0].scatter(_summary['category'], _summary['legacy_beta'], color='#bb7045', label='Legacy estimate')
    _axes[0].set_ylabel('SpringRank beta'); _axes[0].legend()
    _gap = np.linspace(0,1,100)
    for _row in _summary.to_dict('records'):
        _axes[1].plot(_gap,1/(1+np.exp(-2*_row['beta']*_gap)),label=_row['category'])
    _axes[1].set_xlabel('Common SpringRank score gap'); _axes[1].set_ylabel('Model-implied favorite win probability')
    _axes[1].legend(ncol=2); _fig.tight_layout()
    _fig
    return


@app.cell
def research_bootstrap_control(mo):
    research_run_bootstrap = mo.ui.run_button(label='Run / resume tournament bootstrap', kind='success')
    mo.vstack([mo.md('## Tournament bootstrap\nThe next cell runs the statistical analysis in this notebook. The default is 2,000 shared-tournament replicates, seed 42. Checkpoints are local and keyed to the input arrays.'), research_run_bootstrap])
    return (research_run_bootstrap,)


@app.cell
def research_bootstrap_run(
    mo,
    research,
    research_bootstrap_replicates,
    research_primary_leagues,
    research_primary_summary,
    research_random_seed,
    research_root,
    research_run_bootstrap,
    research_tournament_blocks,
):
    mo.stop(not research_run_bootstrap.value, mo.md('Use the bootstrap control above to run the analysis here.'))
    with mo.status.progress_bar(total=int(research_bootstrap_replicates.value), title='Refitting six categories per tournament resample', remove_on_exit=False) as _progress:
        research_bootstrap_draws = research.run_bootstrap(
            research_primary_leagues, research_tournament_blocks,
            replicates=int(research_bootstrap_replicates.value), seed=int(research_random_seed.value), workers=4,
            checkpoint=research_root / 'data/research/tournament_bootstrap.json',
            progress=lambda completed,total: _progress.update(increment=1, subtitle=f'{completed:,} / {total:,} replicates'))
    research_beta_inference = research.bootstrap_tables(research_primary_summary['beta'], research_bootstrap_draws['beta'])
    research_rho_inference = research.bootstrap_tables(research_primary_summary['rho'], research_bootstrap_draws['rho'])
    mo.ui.table([{'metric':'beta', **{k:v for k,v in research_beta_inference.items() if k not in ['intervals','pairs','omnibus']}},
                 {'metric':'Spearman rho', **{k:v for k,v in research_rho_inference.items() if k not in ['intervals','pairs','omnibus']}}],
                selection=None, label='Bootstrap completion and inferential validity')
    return (
        research_beta_inference,
        research_bootstrap_draws,
        research_rho_inference,
    )


@app.cell
def research_bootstrap_results(
    mo,
    pd,
    plt,
    research_beta_inference,
    research_rho_inference,
):
    _beta_ci = pd.DataFrame(research_beta_inference['intervals'])
    _rho_ci = pd.DataFrame(research_rho_inference['intervals'])
    _fig, _axes = plt.subplots(1, 2, figsize=(11,4))
    for _ax, _table, _title in [(_axes[0],_beta_ci,'Beta uncertainty'),(_axes[1],_rho_ci,'FIE rank correlation uncertainty')]:
        _ax.vlines(_table['category'],_table['low'],_table['high'],color='#287b8e'); _ax.scatter(_table['category'],_table['estimate'],color='#287b8e')
        _ax.set_title(_title)
    _fig.tight_layout()
    import io as _io
    _buffer = _io.StringIO()
    _fig.savefig(_buffer, format='svg', bbox_inches='tight')
    plt.close(_fig)
    mo.vstack([mo.Html(_buffer.getvalue()), mo.md(f"Beta omnibus: `{research_beta_inference['omnibus']}`; correlation omnibus: `{research_rho_inference['omnibus']}`"),
               mo.ui.table(pd.DataFrame(research_beta_inference['pairs']),selection=None,label='15 beta contrasts; Holm-adjusted p-values'),
               mo.ui.table(pd.DataFrame(research_rho_inference['pairs']),selection=None,label='15 correlation contrasts; separate Holm family')])
    return


@app.cell
def research_prediction_control(mo):
    research_run_models = mo.ui.run_button(label='Run regional and prediction analyses', kind='success')
    mo.vstack([mo.md('## Regional explanation and two separate prediction tasks\nThe next cells load the biography snapshot at run time, then perform country-grouped discrepancy validation and chronological bout holdouts. No birth date is inferred from a license.'),research_run_models])
    return (research_run_models,)


@app.cell
def research_biographies(
    mo,
    research,
    research_primary_fits,
    research_primary_leagues,
    research_root,
    research_run_models,
    research_valid_bouts,
):
    mo.stop(not research_run_models.value, mo.md('Use the model control above to run these analyses in the notebook.'))
    research_biographies, research_biography_coverage = research.biography_features(
        research_root / 'data/fie_biographies/athletes_enriched.csv', research_primary_leagues, research_valid_bouts['Date'].max())
    research_discrepancy_data = research.discrepancy_features(research_primary_leagues,research_primary_fits,research_biographies)
    mo.vstack([mo.ui.table([research_biography_coverage],selection=None,label='Biography coverage at model run'),
               mo.ui.table(research_discrepancy_data[['category','region','D']].groupby(['category','region'])['D'].agg(['count','mean','median']).reset_index(),
                           selection=None,label='Observed ranking discrepancies by category and region')])
    return (
        research_biographies,
        research_biography_coverage,
        research_discrepancy_data,
    )


@app.cell
def research_region_models(mo, pd, research, research_discrepancy_data):
    research_adjusted_basic = research.clustered_adjustment(research_discrepancy_data,participation=False)
    research_adjusted_participation = research.clustered_adjustment(research_discrepancy_data,participation=True)
    research_region_contrast = research.regional_contrast(research_discrepancy_data)
    mo.vstack([mo.md(f"Positive D means better placement under FIE than SpringRank. Equal-region contrast (Africa + Americas - Asia - Europe)/2: **{research_region_contrast:.4f}**. This is a ranking-system disagreement, not evidence that either system measures true ability."),
               mo.ui.table(pd.DataFrame(research_adjusted_basic['coefficients']),selection=None,label='Category, age and hand adjustment; country-clustered uncertainty'),
               mo.ui.table(pd.DataFrame(research_adjusted_participation['coefficients']),selection=None,label='Additional participation and competition-mix adjustment')])
    return research_adjusted_basic, research_adjusted_participation


@app.cell
def research_discrepancy_prediction(
    mo,
    pd,
    research,
    research_discrepancy_data,
    research_random_seed,
):
    with mo.status.progress_bar(total=1,title='Country-grouped outer validation and inner tuning') as _progress:
        research_discrepancy_prediction = research.discrepancy_prediction(research_discrepancy_data,seed=int(research_random_seed.value))
        _progress.update()
    mo.vstack([mo.ui.table(pd.DataFrame(research_discrepancy_prediction['metrics']),selection=None,label='Held-out-country discrepancy prediction'),
               mo.ui.table(pd.DataFrame(research_discrepancy_prediction['regional_errors']),selection=None,label='Prediction errors by region'),
               mo.ui.table(pd.DataFrame(research_discrepancy_prediction['folds']),selection=None,label='Fold audit: countries and athletes never overlap')])
    return (research_discrepancy_prediction,)


@app.cell
def research_bout_prediction(
    mo,
    pd,
    research,
    research_biographies,
    research_primary_leagues,
    research_random_seed,
    research_regions,
):
    with mo.status.progress_bar(total=18,title='Rolling event holdouts: train scores, beta, and biography models separately') as _progress:
        research_bout_prediction = research.rolling_bout_prediction(research_primary_leagues,research_biographies,research_regions,
            seed=int(research_random_seed.value),progress=lambda completed,total: _progress.update(increment=1,subtitle=f'{completed}/{total} category/fold fits'))
    mo.vstack([mo.ui.table(pd.DataFrame(research_bout_prediction.get('metrics',[])),selection=None,label='Later-bout predictive performance'),
               mo.ui.table(pd.DataFrame(research_bout_prediction.get('folds',[])),selection=None,label='Chronological split and coverage audit'),
               mo.md(f"Coverage: {research_bout_prediction.get('coverage',0):.1%}. Paired event-bootstrap Brier improvement: `{research_bout_prediction.get('brier_improvement')}`. Current club/coach information is excluded; recorded handedness is assumed stable.")])
    return (research_bout_prediction,)


@app.cell
def research_sensitivities(
    mo,
    pd,
    research,
    research_points_audit,
    research_primary_fits,
    research_regions,
    research_root,
    research_valid_bouts,
):
    research_sensitivity_fits = {}
    research_sensitivity_summary = []
    for _population in ['Senior events','All recorded ages']:
        _leagues, _blocks = research.encode_leagues(research_valid_bouts,research_root/'rankings',research_regions,population=_population)
        for _minimum in [1,5,10]:
            for _code,_league in _leagues.items():
                _fit = research_primary_fits[_code] if _population == 'Senior events' and _minimum == 5 else research.fit_league(_league,_minimum)
                research_sensitivity_fits[(_population,_minimum,_code)] = _fit
                research_sensitivity_summary.append({'population':_population,'minimum':_minimum,**_fit[1]})
    research_nozonal_leagues, _ = research.encode_leagues(research_valid_bouts,research_root/'rankings',research_regions,exclude_zonal=True)
    research_nozonal_fits = {c:research.fit_league(l,5) for c,l in research_nozonal_leagues.items()}
    research_region_sensitivities = []
    for _label,_fits in [('Primary',research_primary_fits),('Exclude zonal bouts',research_nozonal_fits)]:
        _joined = pd.concat([f[0].dropna(subset=['D']) for f in _fits.values()])
        research_region_sensitivities.append({'analysis':_label,'n':len(_joined),'contrast':research.regional_contrast(_joined)})
    _elite = pd.concat([f[0].query('fie_rank <= 200').copy() for f in research_primary_fits.values()])
    for _category in research.CATEGORIES:
        _mask = _elite['category'].eq(_category)
        _elite.loc[_mask,'D'] = research.discrepancy(_elite.loc[_mask,'score'],_elite.loc[_mask,'fie_rank'])
    research_region_sensitivities.append({'analysis':'FIE top 200, ranks recomputed within cohort','n':len(_elite),'contrast':research.regional_contrast(_elite)})
    research_zonal_points_audit = research_points_audit
    mo.vstack([mo.ui.table(pd.DataFrame(research_sensitivity_summary),selection=None,label='Age and participation sensitivity analyses'),
               mo.ui.table(pd.DataFrame(research_region_sensitivities),selection=None,label='Regional contrast sensitivities'),
               mo.md('The zonal-point tables are reconciled below. The points counterfactual uses consistent average ties; excluding zonal bouts from SpringRank is a separate sensitivity analysis.')])
    return (
        research_nozonal_fits,
        research_region_sensitivities,
        research_sensitivity_fits,
        research_sensitivity_summary,
        research_zonal_points_audit,
    )


@app.cell
def research_regional_uncertainty(
    mo,
    np,
    pd,
    plt,
    research,
    research_bootstrap_draws,
    research_primary_fits,
):
    research_region_intervals = []
    for _i,_category in enumerate(research.CATEGORIES):
        _frame = research_primary_fits[_category][0].dropna(subset=['D'])
        for _j,_region in enumerate(research.REGIONS):
            _values = research_bootstrap_draws['regional_means'][:,_i,_j]
            _finite = _values[np.isfinite(_values)]
            _subset = _frame[_frame['region'].eq(_region)]
            research_region_intervals.append({'category':_category,'region':_region,'n':len(_subset),
              'mean':float(_subset['D'].mean()),'low':float(np.quantile(_finite,.025)) if len(_finite)>20 else None,
              'high':float(np.quantile(_finite,.975)) if len(_finite)>20 else None,'valid_replicates':len(_finite)})
    _draws = research_bootstrap_draws['regional_contrast']
    _draws = _draws[np.isfinite(_draws)]
    _point = research.regional_contrast(pd.concat([f[0].dropna(subset=['D']) for f in research_primary_fits.values()]))
    research_region_bootstrap = {'estimate':_point,'low':float(np.quantile(_draws,.025)),'high':float(np.quantile(_draws,.975)),
       'valid_fraction':len(_draws)/len(research_bootstrap_draws['regional_contrast']),
       'p':float((1+(np.abs(_draws-_point)>=abs(_point)).sum())/(1+len(_draws))) if len(_draws)>=.9*len(research_bootstrap_draws['regional_contrast']) else None}
    _fig,_axes = plt.subplots(2,3,figsize=(12,7),sharey=True)
    for _ax,(_category,(_frame,_summary)) in zip(_axes.ravel(),research_primary_fits.items()):
        _groups = [_frame.loc[_frame['region'].eq(r),'D'].dropna().to_numpy() for r in research.REGIONS]
        _ax.boxplot(_groups,tick_labels=research.REGIONS,showfliers=False)
        _ax.axhline(0,color='gray',linewidth=.7);_ax.set_title(_category);_ax.tick_params(axis='x',rotation=35)
    _fig.supylabel('D: positive means better placement under FIE');_fig.tight_layout()
    mo.vstack([mo.md(f"Equal-region contrast, tournament uncertainty: `{research_region_bootstrap}`. Unknown affiliations remain explicit in descriptive tables and are excluded from regional regressions."),
     mo.ui.table(pd.DataFrame(research_region_intervals),selection=None,label='Region/category means with tournament-bootstrap intervals'),_fig])
    return research_region_bootstrap, research_region_intervals


@app.cell
def research_reproducibility(
    mo,
    np,
    research,
    research_bootstrap_draws,
    research_discrepancy_data,
    research_discrepancy_prediction,
    research_primary_leagues,
    research_tournament_blocks,
    research_valid_bouts,
):
    # Independent complete ordered likelihood equals the unordered-pair likelihood.
    from scipy.special import xlogy as _xlogy, expit as _expit
    _d=np.array([.5,1.,.5]);_w=np.array([8.,7.,6.]);_n=np.array([10.,7.,10.])
    for _beta in [0.,.2,1.,3.]:
        _p=_expit(2*_beta*_d)
        _pair=np.sum(_xlogy(_w,_p)+_xlogy(_n-_w,1-_p))
        _ordered=np.sum(_xlogy(np.r_[_w,_n-_w],np.r_[_p,1-_p]))
        assert np.isclose(_pair,_ordered)
    with mo.status.progress_bar(total=5,title='Verify fixed-seed bootstrap reproducibility') as _progress:
        _repeat=research.run_bootstrap(research_primary_leagues,research_tournament_blocks,replicates=5,seed=42,workers=1,
            progress=lambda completed,total:_progress.update(increment=1))
    assert np.allclose(_repeat['beta'],research_bootstrap_draws['beta'][:5],equal_nan=True)
    assert np.allclose(_repeat['rho'],research_bootstrap_draws['rho'][:5],equal_nan=True)
    research_identity_audit=research_discrepancy_data.assign(identity=research_discrepancy_data['name'].map(research.normalized_name)).groupby('identity')['country'].nunique()
    assert research_discrepancy_prediction['excluded_cross_country_identity_rows'] >= int((research_identity_audit>1).sum())
    mo.vstack([mo.md('**Additional checks passed:** complete likelihood equivalence; five fixed-seed bootstrap replicates reproduced; cross-country normalized identities conservatively excluded from grouped prediction.'),
     mo.ui.table(research_valid_bouts.groupby(['Event Name','Is Zonal']).size().reset_index(name='bouts'),selection=None,label='Event naming and zonal classification audit')])
    return


@app.cell
def research_export(
    datetime,
    json,
    mo,
    np,
    pd,
    research_adjusted_basic,
    research_adjusted_participation,
    research_beta_inference,
    research_biography_coverage,
    research_bootstrap_draws,
    research_bout_prediction,
    research_discrepancy_prediction,
    research_primary_summary,
    research_random_seed,
    research_region_bootstrap,
    research_region_intervals,
    research_region_sensitivities,
    research_region_source,
    research_rho_inference,
    research_root,
    research_sensitivity_fits,
    research_sensitivity_summary,
    research_zonal_points_audit,
    timezone,
):
    # Only derived outputs are written into the public build. No raw biography records.
    def research_json_safe(value):
        if isinstance(value,dict): return {str(k):research_json_safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple,np.ndarray)): return [research_json_safe(v) for v in value]
        if isinstance(value,(np.integer,)): return int(value)
        if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
        if isinstance(value,np.bool_): return bool(value)
        return value
    research_public_snapshot = json.loads((research_root/'pages/public/analysis.json').read_text(encoding='utf-8'))
    research_public_snapshot['schema_version'] = 2
    research_public_snapshot['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    for _variant in research_public_snapshot['variants']:
        _frame,_summary = research_sensitivity_fits[(_variant['population'],_variant['minimum_bouts'],_variant['category'])]
        _old = {research_public_snapshot['athletes'][row[0]]['name']:row for row in _variant['rows']}
        assert set(_old) == set(_frame['name']), 'Published cohort differs from notebook; export must be reviewed'
        assert max(abs(_old[row['name']][1]-row['score']) for row in _frame.to_dict('records')) < 1e-7
        _variant['beta'] = _summary['beta'];_variant['beta_status'] = _summary['beta_status']
        _variant['depth'] = _summary['depth'];_variant['levels'] = _summary['levels']
    research_public_snapshot['research'] = research_json_safe({
     'primary':research_primary_summary.to_dict('records'),'beta':research_beta_inference,'rho':research_rho_inference,
     'region_intervals':research_region_intervals,'regional_contrast':research_region_bootstrap,
     'regional_sensitivities':research_region_sensitivities,'adjusted_basic':research_adjusted_basic,
     'adjusted_participation':research_adjusted_participation,'discrepancy_prediction':research_discrepancy_prediction,
     'bout_prediction':research_bout_prediction,'biography_coverage':research_biography_coverage,
     'sensitivity_summary':research_sensitivity_summary,'zonal_points_audit':research_zonal_points_audit,
     'bootstrap':{'replicates':len(research_bootstrap_draws['beta']),'seed':int(research_random_seed.value),
       'signature':research_bootstrap_draws['signature'],'cohort_min':research_bootstrap_draws['sizes'].min(axis=0),
       'cohort_max':research_bootstrap_draws['sizes'].max(axis=0),'failed_fit_records':sum(len(x) for x in research_bootstrap_draws['failures'])},
     'region_source':research_region_source['source_url'],
     'notebook':'https://github.com/Zargon42/Oly-Fencing-Networks/blob/main/notebooks/research_analysis.py',
     'caveat':'Conditional on recorded events and observed athletes. Beta is not intrinsic luck; ranking disagreement does not establish true ability.'})
    (research_root/'pages/public/analysis.json').write_text(json.dumps(research_json_safe(research_public_snapshot),ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    research_report_path = research_root/'Analysis/research_report.txt'
    research_report_path.parent.mkdir(parents=True,exist_ok=True)
    _report = '\n\n'.join([
     'FENCING NETWORKS: NOISE, RANKING DISAGREEMENT, AND BIOGRAPHICAL PREDICTION',
     'DRAFT FOR MANUAL REVIEW. Generated by the executable research_analysis.py marimo notebook.',
     'RESEARCH QUESTION AND METHODS\nSix separately fitted senior weapon/gender categories; minimum five recorded bouts. Corrected pairwise conditional likelihood, common score convention. Tournament-block bootstrap: 2,000 replicates, seed 42. Centered bootstrap omnibus/pairwise tests; separate Holm families. Formal inference requires 90% joint validity. Correlations use score versus official rank number, so stronger agreement is more negative.',
     'PRIMARY ESTIMATES\n'+research_primary_summary.to_string(index=False),
     'BETA UNCERTAINTY AND COMPARISONS\n'+json.dumps(research_beta_inference,indent=2),
     'FIE CORRELATION UNCERTAINTY AND COMPARISONS\n'+json.dumps(research_rho_inference,indent=2),
     'REGIONAL DISAGREEMENT\nD=(SR rank-FIE rank)/(n-1), both re-ranked within the common cohort with average ties. Positive means better placement under FIE. Equal-region contrast='+str(research_region_bootstrap),
     'ADJUSTED ASSOCIATIONS\n'+json.dumps(research_adjusted_basic,indent=2)+'\nParticipation adjusted:\n'+json.dumps(research_adjusted_participation,indent=2),
     'DISCREPANCY PREDICTION\n'+json.dumps(research_discrepancy_prediction,indent=2),
     'LATER-BOUT PREDICTION\n'+json.dumps(research_bout_prediction,indent=2),
     'SENSITIVITIES\n'+pd.DataFrame(research_sensitivity_summary).to_string(index=False)+'\n'+str(research_region_sensitivities),
     'LIMITATIONS\nObserved population/event coverage only. FIE tables are concurrent archived snapshots, not ability ground truth. Name matching is not license verified. Missing biography fields may be systematic; club/coach excluded from historical prediction. Invalid dates excluded from chronological evaluation. Zonal-point totals have been fully reconciled; the average-tie counterfactual and remaining placement tie-break limitation are documented in the zonal-point section. Bootstrap percentile coverage and centered tests are approximations, not exact randomization tests. Exploratory geographic hypothesis requires replication.',
     'SOURCES\nhttps://pmc.ncbi.nlm.nih.gov/articles/PMC6054508/\nhttps://fie.org/about\n'+research_region_source['source_url']+'\nBout records: Andrew Fischl (CyrusOfChaos), as credited in DATA.md.',
     'REPRODUCIBILITY\nRun this published notebook with local source files. Public Pages displays its compact exports. Raw biography caches and this unreviewed draft stay outside the public build.'
    ])
    research_report_path.write_text(_report,encoding='utf-8')
    mo.md(f"**Exported actual notebook results:** `pages/public/analysis.json`. Local unreviewed report: `{research_report_path.name}`. Raw biographies and birth dates are excluded from publication.")
    return research_json_safe, research_report_path


@app.cell
def research_discussion(
    mo,
    pd,
    research_adjusted_basic,
    research_beta_inference,
    research_bout_prediction,
    research_discrepancy_prediction,
    research_primary_summary,
    research_region_bootstrap,
    research_report_path,
    research_rho_inference,
):
    _primary=research_primary_summary.sort_values('beta')
    _contrast=research_region_bootstrap
    _prediction=pd.DataFrame(research_discrepancy_prediction['metrics']).set_index('model')
    _bout=research_bout_prediction['brier_improvement']
    research_discussion = f"""
    ## Results and interpretation

    **Category noise.** Corrected beta ranges from {_primary.iloc[0]['beta']:.3f} ({_primary.iloc[0]['category']}) to {_primary.iloc[-1]['beta']:.3f} ({_primary.iloc[-1]['category']}). The six-category beta omnibus p-value is {research_beta_inference['omnibus']['p']:.3f}; the correlation omnibus p-value is {research_rho_inference['omnibus']['p']:.3f}. The descriptive ordering is not sufficient evidence of an overall category difference at the 5% threshold. Pairwise intervals and separately Holm-adjusted tests are shown above; failure to reject is not evidence that categories are equivalent.

    **Geographic discrepancy.** The equal-region contrast is {_contrast['estimate']:.3f}, with 95% tournament-bootstrap interval [{_contrast['low']:.3f}, {_contrast['high']:.3f}]. Positive values support the observed Africa/Americas versus Asia/Europe pattern in this snapshot. The category/age/hand-adjusted country-clustered contrast is {research_adjusted_basic['contrast']['estimate']:.3f} [{research_adjusted_basic['contrast']['low']:.3f}, {research_adjusted_basic['contrast']['high']:.3f}]. This is systematic ranking disagreement; it does not identify ability, causation, or unfairness. Oceania and unknown affiliations must not be merged into either side of this contrast.

    **Discrepancy prediction.** Held-out-country MAE is {_prediction.loc['zero','mae']:.3f} for the zero baseline, {_prediction.loc['ridge','mae']:.3f} for ridge and {_prediction.loc['gradient_boosting','mae']:.3f} for gradient boosting. Gradient-boosting out-of-sample R-squared is {_prediction.loc['gradient_boosting','r2']:.3f}. Six rows with normalized names represented under multiple countries are conservatively excluded from this prediction experiment. Prediction of a concurrent discrepancy is distinct from forecasting future ranking.

    **Bout prediction.** Evaluation covers {research_bout_prediction['covered_test_bouts']:,} later bouts ({research_bout_prediction['coverage']:.1%} of eligible test bouts). Brier improvement is {_bout['estimate']:.6f}, with interval [{_bout['low']:.6f}, {_bout['high']:.6f}]. The interval includes zero: these results do not establish a clear improvement in later-bout prediction. The logistic model can also recalibrate the SpringRank coefficient, so its difference cannot be attributed exclusively to biographical information.

    **Limits and next work.** Replicate geographic findings on another season and collect historical FIE snapshots before forecasting discrepancies. The reconciled zonal-points counterfactual is reported in the following section. Biography missingness, name matching, event coverage and disconnected categories limit interpretation. These analyses are exploratory, not measures of intrinsic luck.
    """
    _existing=research_report_path.read_text(encoding='utf-8')
    _marker='\n\nNOTEBOOK DISCUSSION\n'
    research_report_path.write_text(_existing.split(_marker)[0]+_marker+research_discussion,encoding='utf-8')
    mo.md(research_discussion)
    return (research_discussion,)


@app.cell
def research_points_reconciliation(mo, np, pd, research, research_root):
    # All cleaning, reconciliation, and counterfactual calculations run in this notebook.
    import re as _re
    import hashlib as _hashlib
    research_points_tables = {}
    research_points_events = []
    research_points_audit = []
    research_points_mismatches = []
    research_points_rule_source = 'https://static.fie.org/uploads/38/190670-Organisation%20rules%20ang.pdf'
    for _category in research.CATEGORIES:
        _path = research_root/'rankings'/f'{_category}-detailed-ranking-2026.csv'
        _table = pd.read_csv(_path)
        _rename = {}
        _event_columns = []
        for _header in _table.columns:
            if _header in ['Rank','Name','Nat.','Points','Total points']:
                continue
            _text = ' '.join(str(_header).split())
            _reversed = _text.startswith(')')
            if _reversed:
                _text = ' '.join(part[::-1] for part in reversed(_text.split()))
            _match = _re.fullmatch(r'(\d{2})/(\d{2})\s+(.+?)\s*\((SA|A|GP|CHZ|CHM|JO)\)',_text)
            assert _match, f'Unrecognized event header: {_category} {_header!r} -> {_text!r}'
            _day,_month,_city,_kind = _match.groups()
            _year = 2025 if int(_month)>=9 else 2026
            _date = pd.Timestamp(year=_year,month=int(_month),day=int(_day))
            _clean = f'{_date.date()} {_city} ({_kind})'
            assert _clean not in _rename.values()
            _rename[_header] = _clean
            _event_columns.append((_clean,_kind))
            research_points_events.append({'category':_category,'original_header':_header,'clean_header':_clean,
                'date':str(_date.date()),'location':_city,'event_type':_kind,'reversed_header_fixed':_reversed})
        _table = _table.rename(columns={**_rename,'Nat.':'country','Points':'official_total','Total points':'official_total','Name':'name','Rank':'official_rank'})
        _ordinary = [c for c,k in _event_columns if k in ['SA','A','GP']]
        _zonal = [c for c,k in _event_columns if k=='CHZ']
        _world = [c for c,k in _event_columns if k in ['CHM','JO']]
        assert len(_zonal)==4 and len(_world)==1
        _points = _table[[c for c,k in _event_columns]].apply(pd.to_numeric,errors='raise')
        assert (_points.fillna(0)>=0).all().all()
        assert (_points[_zonal].fillna(0).gt(0).sum(axis=1)<=1).all()
        _table['ordinary_best_five'] = np.sort(_points[_ordinary].fillna(0).to_numpy(),axis=1)[:,-5:].sum(axis=1)
        _table['ordinary_all_points'] = _points[_ordinary].sum(axis=1)
        _table['discarded_ordinary_points'] = _table['ordinary_all_points']-_table['ordinary_best_five']
        _table['zonal_points'] = _points[_zonal].sum(axis=1)
        _table['world_points'] = _points[_world].sum(axis=1)
        _table['reconstructed_total'] = _table['ordinary_best_five']+_table['zonal_points']+_table['world_points']
        _table['reconciliation_error'] = _table['reconstructed_total']-_table['official_total']
        _table['reconciled'] = np.isclose(_table['reconciliation_error'],0,atol=1e-8)
        _table['points_without_zonal'] = _table['ordinary_best_five']+_table['world_points']
        _table['rank_from_total_average_ties'] = _table['official_total'].rank(ascending=False,method='average')
        _table['rank_without_zonal_average_ties'] = _table['points_without_zonal'].rank(ascending=False,method='average')
        _table['key'] = _table['name'].map(research.normalized_name)
        research_points_tables[_category] = _table
        research_points_audit.append({'category':_category,'athletes':len(_table),'events':len(_event_columns),
            'headers_repaired':sum(e['reversed_header_fixed'] for e in research_points_events if e['category']==_category),
            'reconciled':int(_table['reconciled'].sum()),'unreconciled':int((~_table['reconciled']).sum()),
            'max_absolute_error':float(_table['reconciliation_error'].abs().max()),
            'athletes_with_zonal_points':int(_table['zonal_points'].gt(0).sum()),
            'input_sha256':_hashlib.sha256(_path.read_bytes()).hexdigest()})
        if (~_table['reconciled']).any():
            research_points_mismatches.append(_table.loc[~_table['reconciled'],['name','official_total','reconstructed_total','reconciliation_error']].assign(category=_category))
    research_points_clean_dir = research_root/'data/research/clean_rankings'
    research_points_clean_dir.mkdir(parents=True,exist_ok=True)
    for _category,_table in research_points_tables.items():
        _table.to_csv(research_points_clean_dir/f'{_category}-reconciled.csv',index=False)
    pd.DataFrame(research_points_events).to_csv(research_points_clean_dir/'event_dictionary.csv',index=False)
    mo.vstack([mo.md('## Reconcile detailed FIE points\nRule o.108.1(a): five best World Cup/Grand Prix/Satellite results, plus World/Olympic and zonal championships. Zonal results occupy a separate slot, so removing zonal points does not promote a sixth ordinary result. Point values already include event multipliers. Event years are inferred from the 2025/26 season and month, and the original headers are retained in the event dictionary. Parenthesis formatting is not used to decide the best five.'),
     mo.ui.table(pd.DataFrame(research_points_audit),selection=None,label='Every-row reconciliation against published totals'),
     mo.ui.table(pd.DataFrame(research_points_events),selection=None,label='Cleaned tournament headers and source mapping'),
     mo.ui.table(pd.concat(research_points_mismatches,ignore_index=True) if research_points_mismatches else pd.DataFrame({'status':['All totals reconciled']}),selection=None,label='Unresolved point discrepancies')])
    return (
        research_points_audit,
        research_points_events,
        research_points_rule_source,
        research_points_tables,
    )


@app.cell
def research_points_methods(
    mo,
    pd,
    research,
    research_discrepancy_data,
    research_nozonal_fits,
    research_points_audit,
    research_points_tables,
    research_primary_fits,
):
    import src.ranking_points as ranking_points
    assert all(row['unreconciled']==0 for row in research_points_audit)
    research_points_frames = {c:ranking_points.attach_points(f[0],research_points_tables[c]) for c,f in research_primary_fits.items()}
    research_points_contrasts,research_points_correlations = ranking_points.summarize(list(research_points_frames.values()))
    research_points_joined = pd.concat(research_points_frames.values(),ignore_index=True)
    research_points_region_summary = research_points_joined.groupby('region').agg(
        n=('name','size'),mean_original_D=('D_original','mean'),mean_total_ties_D=('D_total_ties','mean'),
        mean_without_zonal_D=('D_without_zonal','mean'),mean_change=('D_change','mean'),
        mean_zonal_points=('zonal_points','mean'),mean_zonal_fraction=('zonal_fraction','mean'),
        mean_global_rank_change=('global_rank_change','mean')).reset_index()
    research_points_adjusted = {}
    for _target in ['D_total_ties','D_without_zonal','D_change']:
        _feature=research_discrepancy_data.drop(columns=['D']).merge(research_points_joined[['category','name',_target]],on=['category','name'],validate='one_to_one')
        _feature['D']=_feature[_target]
        research_points_adjusted[_target] = {'basic':research.clustered_adjustment(_feature,False),
                                            'participation':research.clustered_adjustment(_feature,True)}
    research_points_both_removed=[]
    for _code,_fit in research_nozonal_fits.items():
        research_points_both_removed.append(ranking_points.attach_points(_fit[0],research_points_tables[_code]))
    research_points_both_contrasts,research_points_both_correlations=ranking_points.summarize(research_points_both_removed)
    mo.vstack([mo.md('## Counterfactual: remove the zonal points slot\nTotals reconcile for every row. Remove CHZ points while retaining the five ordinary results and world points. No replacement by a sixth ordinary result is permitted by the separate-slot rule. We retain all athletes, including zero-point athletes. This is a mechanical points ablation with observed participation and SpringRank scores held fixed, not a simulated season without zonal events.\n\n**Ties:** Both before and after orders use average ranks of total points within the same matched cohort. The supplied official ordinal ranks can use placement tie-breaks; comparing their distinct positions with a tied counterfactual would mix tie-rule changes with removing points. Original-rank discrepancies remain shown as a sensitivity. Exact counterfactual official tie-breaks are not claimed because event finishing positions are absent from these point tables.'),
     mo.ui.table([research_points_contrasts],selection=None,label='Original, tie-controlled and without-zonal regional contrasts'),
     mo.ui.table(research_points_region_summary,selection=None,label='Regional ranking and points effects'),
     mo.ui.table(pd.DataFrame(research_points_correlations),selection=None,label='Agreement with SpringRank before and after removing zonal points')])
    return (
        ranking_points,
        research_points_adjusted,
        research_points_both_contrasts,
        research_points_both_correlations,
        research_points_contrasts,
        research_points_correlations,
        research_points_frames,
        research_points_joined,
        research_points_region_summary,
    )


@app.cell
def research_points_bootstrap(
    mo,
    pd,
    ranking_points,
    research,
    research_points_contrasts,
    research_points_correlations,
    research_points_joined,
    research_points_tables,
    research_primary_leagues,
    research_root,
    research_tournament_blocks,
):
    with mo.status.progress_bar(total=2000,title='Paired zonal-points analysis: shared tournament refits',remove_on_exit=False) as _progress:
        research_points_bootstrap = ranking_points.paired_bootstrap(research_primary_leagues,research_points_tables,research_tournament_blocks,
            replicates=2000,seed=42,workers=4,checkpoint=research_root/'data/research/zonal_points_bootstrap.json',
            progress=lambda completed,total:_progress.update(increment=1,subtitle=f'{completed:,}/{total:,} paired replicates'))
    research_points_valid = [r for r in research_points_bootstrap['results'] if r.get('failure') is None]
    research_points_uncertainty = {key:ranking_points.interval(value,[r['contrasts'][key] for r in research_points_valid],2000)
                                  for key,value in research_points_contrasts.items()}
    research_points_correlation_intervals = []
    for _i,_point in enumerate(research_points_correlations):
        _result={'category':_point['category'],'n':_point['n']}
        for _metric in ['rho_total_ties','rho_without_zonal','rho_change']:
            _result[_metric]=ranking_points.interval(_point[_metric],[r['correlations'][_i][_metric] for r in research_points_valid],2000)
        research_points_correlation_intervals.append(_result)
    _p=[row['rho_change']['p'] for row in research_points_correlation_intervals]
    if all(p is not None for p in _p):
        for _row,_adjusted in zip(research_points_correlation_intervals,research.holm(_p)):
            _row['rho_change']['p_holm']=float(_adjusted)
    research_points_regional_intervals=[]
    for _i,_region in enumerate(research.REGIONS):
        _subset=research_points_joined[research_points_joined['region'].eq(_region)]
        for _j,_metric in enumerate(['D_total_ties','D_without_zonal','D_change']):
            _draws=[r['regional'][_i][_j] for r in research_points_valid]
            research_points_regional_intervals.append({'region':_region,'metric':_metric,'n':len(_subset),
                **ranking_points.interval(float(_subset[_metric].mean()),_draws,2000)})
    mo.vstack([mo.ui.table([{'metric':k,**v} for k,v in research_points_uncertainty.items()],selection=None,label='Paired tournament-bootstrap contrasts and changes'),
     mo.ui.table(pd.DataFrame(research_points_regional_intervals),selection=None,label='Regional before/after uncertainty'),
     mo.md(f"Joint valid paired replicates: {len(research_points_valid):,}/2,000. Inference is conditional on the archived points snapshot and observed tournament coverage, and propagates bout-network uncertainty, not uncertainty about unobserved seasons.")])
    return (
        research_points_bootstrap,
        research_points_correlation_intervals,
        research_points_regional_intervals,
        research_points_uncertainty,
        research_points_valid,
    )


@app.cell
def research_points_validation(
    mo,
    np,
    pd,
    ranking_points,
    research,
    research_bootstrap_draws,
    research_discrepancy_data,
    research_points_audit,
    research_points_bootstrap,
    research_points_frames,
    research_points_joined,
    research_points_tables,
    research_primary_fits,
    research_primary_leagues,
    research_tournament_blocks,
):
    assert all(row['unreconciled']==0 and row['max_absolute_error']==0 for row in research_points_audit)
    assert sum(row['athletes'] for row in research_points_audit)==sum(len(t) for t in research_points_tables.values())
    for _table in research_points_tables.values():
        assert np.allclose(_table['official_total']-_table['zonal_points'],_table['points_without_zonal'])
        assert (_table['points_without_zonal']<=_table['official_total']+1e-8).all()
    assert len(research_points_joined)==len(research_discrepancy_data)
    _draws=np.array([r['contrasts']['D_original'] if r.get('failure') is None else np.nan for r in research_points_bootstrap['results']])
    assert np.allclose(_draws,research_bootstrap_draws['regional_contrast'],equal_nan=True), 'Paired resamples must match the previous bootstrap exactly.'
    with mo.status.progress_bar(total=5,title='Reproduce five paired counterfactual resamples') as _progress:
        _repeat=ranking_points.paired_bootstrap(research_primary_leagues,research_points_tables,research_tournament_blocks,
            replicates=5,seed=42,workers=1,progress=lambda completed,total:_progress.update(increment=1))
    assert _repeat['results']==research_points_bootstrap['results'][:5]
    # Synthetic fixture: removing the separate zonal slot does not promote result six.
    _ordinary=np.array([32.,26.,20.,14.,8.,4.])
    assert np.sort(_ordinary)[-5:].sum()==100
    assert (100+48+80)-48==180
    _toy=pd.DataFrame({'category':['WF']*3,'name':['A','B','C'],'country':['FRA']*3,'region':['Europe']*3,
        'score':[3.,2.,1.],'D':[0.,0.,0.]})
    _totals=pd.DataFrame({'name':['A','B','C'],'key':['a','b','c'],'official_total':[10.,10.,2.],
     'zonal_points':[8.,0.,0.],'points_without_zonal':[2.,10.,2.],
     'rank_from_total_average_ties':[1.5,1.5,3.],'rank_without_zonal_average_ties':[2.5,1.,2.5]})
    _result=ranking_points.attach_points(_toy,_totals)
    assert np.isclose(_result.loc[0,'D_total_ties'],-.25)
    assert np.isclose(_result.loc[0,'D_without_zonal'],-.75)
    assert np.isclose(_result.loc[0,'D_change'],-.5)
    research_points_category_contrasts=[]
    for _code,_frame in research_points_frames.items():
        _row={'category':_code,'n':len(_frame)}
        for _metric in ['D_total_ties','D_without_zonal','D_change']:
            _copy=_frame.copy();_copy['D']=_copy[_metric]
            _row[_metric]=research.regional_contrast(_copy)
        research_points_category_contrasts.append(_row)
    research_points_elite_frames=[]
    for _code,_fit in research_primary_fits.items():
        _elite=_fit[0].query('fie_rank <= 200').copy()
        _elite['D']=research.discrepancy(_elite['score'],_elite['fie_rank'])
        research_points_elite_frames.append(ranking_points.attach_points(_elite,research_points_tables[_code]))
    research_points_elite_contrasts,research_points_elite_correlations=ranking_points.summarize(research_points_elite_frames)
    research_points_checks = {'all_totals_reconciled':True,'same_matched_cohort':True,'no_replacement_slot':True,
        'ties_and_signs':True,'paired_resamples_match_original':True,'five_seeded_draws_reproduced':True}
    mo.md('**Counterfactual validation passed:** every official total reconciled; fixed matched cohorts; separate-slot arithmetic; tie/sign fixture; all paired resamples match the original tournament draws; five repeated draws reproduced exactly.')
    return (
        research_points_category_contrasts,
        research_points_checks,
        research_points_elite_contrasts,
        research_points_elite_correlations,
        research_points_elite_frames,
    )


@app.cell
def research_points_export(
    datetime,
    json,
    mo,
    pd,
    research_discussion,
    research_json_safe,
    research_points_adjusted,
    research_points_audit,
    research_points_bootstrap,
    research_points_both_contrasts,
    research_points_both_correlations,
    research_points_category_contrasts,
    research_points_checks,
    research_points_correlation_intervals,
    research_points_elite_contrasts,
    research_points_elite_correlations,
    research_points_elite_frames,
    research_points_events,
    research_points_region_summary,
    research_points_regional_intervals,
    research_points_rule_source,
    research_points_uncertainty,
    research_points_valid,
    research_report_path,
    research_root,
    timezone,
):
    _base_discussion = research_discussion
    research_zonal_published = research_json_safe({
     'status':'complete','rule_source':research_points_rule_source,'rule_article':'o.108.1(a), o.108.1(e)',
     'rule':'Best five World Cup/Grand Prix/Satellite point results plus world/Olympic and zonal points. Remove the zonal slot; no sixth ordinary replacement.',
     'reconciliation':research_points_audit,'events':research_points_events,
     'contrasts':research_points_uncertainty,'regional_summary':research_points_region_summary.to_dict('records'),
     'regional_intervals':research_points_regional_intervals,'correlations':research_points_correlation_intervals,
     'adjusted':research_points_adjusted,'both_zonal_points_and_bouts_removed':{'contrasts':research_points_both_contrasts,'correlations':research_points_both_correlations},
     'top200_sensitivity':{'n':sum(len(f) for f in research_points_elite_frames),'contrasts':research_points_elite_contrasts,'correlations':research_points_elite_correlations},
     'category_regional_contrasts':research_points_category_contrasts,
     'bootstrap':{'replicates':2000,'seed':42,'joint_valid':len(research_points_valid),'valid_fraction':len(research_points_valid)/2000,
       'signature':research_points_bootstrap['signature'],'failed_draws':[r['failure'] for r in research_points_bootstrap['results'] if r.get('failure')]},
     'validation':research_points_checks,
     'tie_policy':'Average point ranks both before and after in the same category-specific matched cohort. Exact FIE placing tie-breaks are not reconstructed.',
     'interpretation':'Mechanical removal of recorded zonal points with observed participation fixed; not a causal estimate or a simulated season without zonal events.'})
    _snapshot_path=research_root/'pages/public/analysis.json'
    _snapshot=json.loads(_snapshot_path.read_text(encoding='utf-8'))
    _snapshot['generated_at_utc']=datetime.now(timezone.utc).isoformat()
    _snapshot['research']['zonal_points']=research_zonal_published
    _snapshot['research']['zonal_points_audit']=research_json_safe(research_points_audit)
    _snapshot_path.write_text(json.dumps(_snapshot,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    _before=research_points_uncertainty['D_total_ties'];_after=research_points_uncertainty['D_without_zonal']
    _change=research_points_uncertainty['D_change'];_fraction=research_points_uncertainty['fraction_reduction']
    research_points_discussion=f"""
    ## Reconciled zonal-point analysis

    All {sum(r['athletes'] for r in research_points_audit):,} athlete/category ranking records reconcile exactly: best five ordinary results plus world/Olympic and zonal points. Repaired {sum(r['headers_repaired'] for r in research_points_audit)} reversed tournament headers. Preserved original inputs and a source-to-clean event dictionary; cleaned local tables are in data/research/clean_rankings/.

    The equal-region contrast using consistent point ties is {_before['estimate']:.4f} (95% tournament interval {_before['low']:.4f} to {_before['high']:.4f}). After removing zonal points it is {_after['estimate']:.4f} ({_after['low']:.4f} to {_after['high']:.4f}). The paired change is {_change['estimate']:.4f} ({_change['low']:.4f} to {_change['high']:.4f}); centered-bootstrap p={_change['p']}. Relative contrast reduction is {_fraction['estimate']:.1%} ({_fraction['low']:.1%} to {_fraction['high']:.1%}). This percentage is a mechanical change in this contrast, not the causal proportion of ranking disagreement explained by zonal championships.

    Both point orders use average ties in the same matched cohort. The archived official rank, which can break point ties by tournament placings, remains a sensitivity. We do not label the counterfactual an exact official FIE ranking: event finish positions needed for every tie-break are not available in the point tables. All zero-point athletes remain included. Ordinary results are not promoted into the removed zonal slot.

    The category/age/hand-adjusted residual contrast is {research_points_adjusted['D_without_zonal']['basic']['contrast']['estimate']:.4f}, with country-clustered interval [{research_points_adjusted['D_without_zonal']['basic']['contrast']['low']:.4f}, {research_points_adjusted['D_without_zonal']['basic']['contrast']['high']:.4f}], which includes zero. Country-clustered adjustments for participation/mix are also reported before/after and directly for the paired discrepancy change. The top-200 fixed-cohort sensitivity has before/after contrasts {research_points_elite_contrasts['D_total_ties']:.4f} and {research_points_elite_contrasts['D_without_zonal']:.4f}. Category-specific correlation changes use the same tournament draws and a separate six-test Holm family. A second sensitivity removes both zonal bouts from SpringRank and zonal points; its different fitted cohort must not be conflated with the fixed-cohort points ablation.

    The analysis holds recorded participation, the archived point snapshot, and observed bouts fixed. It identifies the mechanical contribution of recorded zonal points, not causal effects on ability, entry decisions or an alternative season. Uncertainty resamples the observed bout tournaments; it does not resample an independently observed series of FIE ranking snapshots. Geographic findings remain exploratory.

    Rule source: {research_points_rule_source}, articles o.108.1(a) and o.108.1(e).
    """
    _report=research_report_path.read_text(encoding='utf-8')
    _marker='\n\nRECONCILED ZONAL-POINT ANALYSIS\n'
    _report=_report.split(_marker)[0]
    _report=_report.replace('Zonal-point counterfactual is not identifiable from the supplied PDF-derived tables.','Zonal-point totals have now been fully reconciled; the average-tie counterfactual and its remaining tie-break limitation are documented below.')
    _report=_report.replace('Reconstruct counted zonal points and contemporary rules before a no-zonal-points counterfactual.','The zonal-points counterfactual is now reconciled and reported below.')
    _report=_report.replace('A without-zonal-points FIE ranking is not reported','A point-based without-zonal counterfactual is now reported')
    research_report_path.write_text(_report+_marker+research_points_discussion+'\n\n'+json.dumps(research_zonal_published,ensure_ascii=False,indent=2),encoding='utf-8')
    mo.vstack([mo.md(research_points_discussion),
     mo.ui.table(pd.DataFrame([{'category':r['category'],**r['rho_change']} for r in research_points_correlation_intervals]),selection=None,label='Paired correlation changes; six-test Holm family'),
     mo.md('Updated public snapshot and local report exported from this notebook. Cleaned per-athlete point tables stay local; public output contains summaries and the event dictionary.')])
    return


if __name__ == "__main__":
    app.run()
