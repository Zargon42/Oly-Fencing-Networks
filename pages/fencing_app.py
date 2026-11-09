# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.23.10",
#     "pandas>=2.2,<3",
#     "matplotlib>=3.9,<4",
# ]
# ///
"""Interactive viewer of actual fencing analysis outputs, suitable for GitHub Pages."""
import marimo

__generated_with = '0.23.10'
app = marimo.App(width='medium', app_title='Fencing network rankings')


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json
    import io
    import sys
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    return Path, io, json, pd, plt, sys


@app.cell
async def _(Path, json, mo, sys):
    if sys.platform == 'emscripten':
        from pyodide.http import pyfetch
        _response = await pyfetch(str(mo.notebook_location() / 'public' / 'analysis.json'))
        if not _response.ok:
            raise RuntimeError('The published analysis snapshot could not be loaded. Please reload the page.')
        snapshot = await _response.json()
    else:
        snapshot = json.loads((Path(mo.notebook_location()) / 'public' / 'analysis.json').read_text(encoding='utf-8'))
    if snapshot.get('schema_version') != 3 or snapshot.get('data_kind') != 'actual_research_outputs':
        raise ValueError('Expected a supported snapshot of actual research outputs.')
    return (snapshot,)


@app.cell(hide_code=True)
def _(mo, snapshot):
    mo.md(f"""
    # How unpredictable is fencing?

    Explore rankings inferred from **actual 2025/26 fencing bouts** using SpringRank.
    Compare the ordering with archived FIE standings and inspect how the results change
    with the athlete population and minimum participation threshold.

    This is a saved analysis snapshot, generated **{snapshot['generated_at_utc'][:10]}**.
    The statistical analysis runs in the [published research notebook](https://github.com/Zargon42/Oly-Fencing-Networks/blob/main/notebooks/research_analysis.py). This app explores its saved results.
    """)
    return


@app.cell
def _(mo, snapshot):
    population = mo.ui.dropdown(options=['Senior events', 'All recorded ages'], value='Senior events', label='Event population', allow_select_none=False)
    category = mo.ui.dropdown(options={label: code for code, label in snapshot['category_labels'].items()}, value="Women's foil", label='Category', allow_select_none=False)
    minimum_bouts = mo.ui.dropdown(options={'1 recorded bout': 1, '5 recorded bouts': 5, '10 recorded bouts': 10}, value='5 recorded bouts', label='Minimum participation', allow_select_none=False)
    elite_only = mo.ui.checkbox(value=False, label='Compare only athletes in the FIE top 200')
    athlete_search = mo.ui.text(label='Search athlete', placeholder='Name or country code', debounce=False)
    top_count = mo.ui.slider(start=5, stop=40, step=5, value=15, label='Athletes in score chart')
    rank_controls = mo.vstack([mo.hstack([population, category, minimum_bouts], justify='start', wrap=True), elite_only])
    return athlete_search, category, elite_only, minimum_bouts, population, top_count, rank_controls


@app.cell
def _(category, minimum_bouts, pd, population, snapshot):
    selected_variant = next(v for v in snapshot['variants'] if v['population'] == population.value and v['category'] == category.value and v['minimum_bouts'] == minimum_bouts.value)
    rankings = pd.DataFrame(selected_variant['rows'], columns=snapshot['row_columns'])
    rankings['athlete'] = rankings['athlete_id'].map(lambda i: snapshot['athletes'][i]['name'])
    rankings['country'] = rankings['athlete_id'].map(lambda i: snapshot['athletes'][i]['country'])
    rankings['fie_rank'] = pd.to_numeric(rankings['fie_rank'], errors='coerce').astype('Int64')
    rankings = rankings.sort_values(['score', 'athlete'], ascending=[False, True])
    return rankings, selected_variant


@app.cell(hide_code=True)
def _(mo, rankings, selected_variant):
    _matched_count = int(rankings['fie_rank'].notna().sum())
    rank_view_6 = mo.md(f"""
    **{len(rankings):,} athletes** · **{selected_variant['retained_bouts']:,} retained bouts** ·
    **{_matched_count:,} matched to archived FIE ranks** ({_matched_count / len(rankings):.1%} coverage)

    Higher SpringRank scores indicate stronger observed performance. Scores are centered separately
    within each fitted category and population; they do not support cross-category match predictions.
    """)
    return rank_view_6


@app.cell
def _(elite_only, rankings):
    comparison = rankings.dropna(subset=['fie_rank']).copy()
    if elite_only.value:
        comparison = comparison[comparison['fie_rank'].le(200)]
    rank_correlation = None
    if len(comparison) >= 3:
        rank_correlation = float(comparison['score'].rank().corr(comparison['fie_rank'].astype(float).rank()))
    return comparison, rank_correlation


@app.cell(hide_code=True)
def _(comparison, elite_only, mo, rank_correlation):
    _scope = 'the matched FIE top 200' if elite_only.value else 'all matched athletes'
    _stat = f'{rank_correlation:.3f}' if rank_correlation is not None else 'not enough matched athletes'
    rank_view_8 = mo.md(f"""
    ## Agreement with FIE standings

    **Spearman correlation: {_stat}**, using **{len(comparison):,} athletes** in {_scope}.
    A negative correlation is expected: the strongest model scores should correspond to the
    smallest official rank numbers. Changing the elite filter changes the comparison sample.
    """)
    return rank_view_8


@app.cell
def _(comparison, io, mo, plt):
    if len(comparison) >= 3:
        _fig, _ax = plt.subplots(figsize=(8, 4.5))
        _ax.scatter(comparison['score'], comparison['fie_rank'].astype(float), s=16, alpha=.5, color='#287b8e', edgecolors='none')
        _ax.set_xlabel('SpringRank score (higher is stronger)')
        _ax.set_ylabel('Archived FIE rank (smaller is better)')
        _ax.invert_yaxis()
        _ax.spines[['top', 'right']].set_visible(False)
        _ax.grid(alpha=.15)
        _fig.tight_layout()
        _buffer = io.StringIO()
        _fig.savefig(_buffer, format='svg', bbox_inches='tight')
        _svg = _buffer.getvalue().replace('<svg ', '<svg style="max-width:100%;height:auto" role="img" aria-label="SpringRank score versus archived FIE rank" ', 1)
        _plot = mo.Html(_svg)
        plt.close(_fig)
    else:
        _plot = mo.md('No FIE comparison is available for this selection.')
    rank_view_9 = _plot
    return rank_view_9


@app.cell(hide_code=True)
def _(athlete_search, mo, top_count):
    rank_view_10 = mo.vstack([mo.md('## Explore the calculated rankings'), mo.hstack([athlete_search, top_count], justify='start', wrap=True)])
    return rank_view_10


@app.cell
def _(athlete_search, rankings):
    _query = athlete_search.value.strip().casefold()
    visible_rankings = rankings
    if _query:
        visible_rankings = rankings[rankings['athlete'].str.casefold().str.contains(_query, regex=False) | rankings['country'].fillna('').str.casefold().str.contains(_query, regex=False)]
    return (visible_rankings,)


@app.cell
def _(io, mo, plt, top_count, visible_rankings):
    if not visible_rankings.empty:
        _top = visible_rankings.head(top_count.value).iloc[::-1]
        _fig, _ax = plt.subplots(figsize=(8, max(3, len(_top) * .25)))
        _ax.barh(_top['athlete'], _top['score'], color='#287b8e')
        _ax.set_xlabel('SpringRank score')
        _ax.spines[['top', 'right']].set_visible(False)
        _fig.tight_layout()
        _buffer = io.StringIO()
        _fig.savefig(_buffer, format='svg', bbox_inches='tight')
        _svg = _buffer.getvalue().replace('<svg ', '<svg style="max-width:100%;height:auto" role="img" aria-label="Highest calculated scores in the current athlete search" ', 1)
        _chart = mo.Html(_svg)
        plt.close(_fig)
    else:
        _chart = mo.md('No athletes match that search.')
    rank_view_12 = _chart
    return rank_view_12


@app.cell
def _(category, minimum_bouts, mo, population, visible_rankings):
    _table = visible_rankings[['model_rank', 'athlete', 'country', 'score', 'retained_bouts', 'fie_rank']].rename(columns={
        'model_rank': 'Model rank', 'athlete': 'Athlete', 'country': 'Country', 'score': 'SpringRank score',
        'retained_bouts': 'Retained bouts', 'fie_rank': 'Archived FIE rank'})
    rank_view_13 = mo.vstack([
        mo.ui.table(_table, pagination=True, page_size=15, selection=None, show_download=False, label='Calculated athlete rankings'),
        mo.download(data=_table.to_csv(index=False).encode('utf-8'), filename=f"{category.value}-{population.value.replace(' ', '-')}-min{minimum_bouts.value}.csv", mimetype='text/csv', label='Download these rankings'),
    ])
    return rank_view_13


@app.cell(hide_code=True)
def _(mo, snapshot):
    _audit = snapshot['audit']
    rank_view_14 = mo.accordion({
        'Methods and scope': mo.md(f"""
        **Inputs:** {_audit['raw_bouts']:,} recorded pool and direct-elimination rows;
        {_audit['fenced_valid_bouts']:,} valid fenced outcomes before category and participation filters.
        Parsable event dates run from {_audit['first_parseable_date']} to {_audit['last_parseable_date']}.
        {_audit['invalid_date_rows']:,} rows have invalid dates and remain eligible for this non-temporal analysis.

        Each weapon/gender category is fitted independently. Senior events excludes junior and cadet events;
        All recorded ages combines those age categories. Mixed-gender events are excluded.
        Participation is counted before removing ineligible opponents, followed by a largest-component filter.
        Consequently, retained bout counts may be below the initial eligibility threshold.
        Pool and direct-elimination wins have equal weight. Scores solve the SpringRank Laplacian objective.

        Athlete identity in bout networks uses recorded names. FIE validation matches unique normalized names
        within the corresponding category; it is not license-verified. Country is the most frequently recorded
        affiliation. Archived FIE ranks describe the supplied 2025/26 tables, not today's live standings.
        """),
        'How to interpret the comparison': mo.md("""
        SpringRank and FIE standings can reflect the same events. Their correlation measures concurrent
        agreement, not future-bout accuracy or proof that one ranking is better. The minimum-participation
        settings change both the fitted network and the athletes included, so stronger correlations under
        a stricter threshold cannot be attributed solely to better estimation. Scores from separate fits
        are not a common scale of ability. Cross-category fantasy probabilities are hypothetical entertainment conventions.
        """),
        'Sources and reproducibility': mo.md("""
        Analysis: [Oly-Fencing-Networks](https://github.com/Zargon42/Oly-Fencing-Networks).
        Bout dataset credited to Andrew Fischl (CyrusOfChaos). SpringRank utilities were adapted from
        Caterina De Bacco's [Probabilistic Inference in Networks material](https://github.com/cdebacco/ds_prob_inf_net).
        Official ranking source: [FIE](https://fie.org/athletes).

        The site contains a compact, actual-results snapshot and its input hashes. It does not contain
        raw event files or biography caches. The published `notebooks/research_analysis.py` exports
        the actual results from visible notebook cells. Browser controls explore the saved fits without fetching
        FIE profiles. Internet access is needed initially for the Python browser runtime and packages.
        """),
    })
    return rank_view_14



@app.cell
def _(io, mo, plt):
    import math
    def public_svg(fig,label):
        buffer=io.StringIO();fig.savefig(buffer,format='svg',bbox_inches='tight');plt.close(fig)
        return mo.Html(buffer.getvalue().replace('<svg ',f'<svg style="max-width:100%;height:auto" role="img" aria-label="{label}" ',1))
    def fantasy_chance(a,ba,b,bb,ma=0.,mb=0.):
        z=2*ba*(a-ma)-2*bb*(b-mb)
        return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
    return public_svg, fantasy_chance


@app.cell
def _(mo,pd,plt,public_svg,snapshot):
    evidence=snapshot['research']
    _beta_test = f"p = {evidence['beta']['omnibus']['p']:.3f}" if evidence['beta'].get('omnibus') else 'formal inference suppressed'
    _rho_test = f"p = {evidence['rho']['omnibus']['p']:.3f}" if evidence['rho'].get('omnibus') else 'formal inference suppressed'
    _fig,_axes=plt.subplots(1,2,figsize=(10,4))
    for _ax,_key,_title in [(_axes[0],'beta','Model-estimated beta'),(_axes[1],'rho','FIE correlation')]:
        _table=pd.DataFrame(evidence[_key]['intervals'])
        _ax.vlines(_table['category'],_table['low'],_table['high'],color='#287b8e')
        _ax.scatter(_table['category'],_table['estimate'],color='#287b8e');_ax.set_title(_title)
    _fig.tight_layout()
    _curves,_ax=plt.subplots(figsize=(8,3.5))
    for _row in evidence['primary']:
        _gaps=[i/100 for i in range(101)]
        _ax.plot(_gaps,[1/(1+__import__('math').exp(-2*_row['beta']*g)) for g in _gaps],label=_row['category'])
    _ax.set(xlabel='Common SpringRank score gap',ylabel='Favorite win probability');_ax.legend(ncol=3)
    noise_panel=mo.vstack([
      mo.md('## Noise across categories\nHigher beta means a sharper relationship between score differences and winning. Lower beta means greater model-estimated noise at a common score gap. This is conditional on observed events, not intrinsic luck. The score-fitting convention is held constant.'),
      public_svg(_fig,'Beta and FIE correlation with 95 percent tournament bootstrap intervals'),
      public_svg(_curves,'Win probability across a common score gap'),
      mo.md(f"**Bootstrap:** {evidence['bootstrap']['replicates']:,} joint tournament resamples, seed {evidence['bootstrap']['seed']}. Valid beta draws: {evidence['beta']['valid_fraction']:.1%}; correlation: {evidence['rho']['valid_fraction']:.1%}. Formal tests require 90%.\n\n**Overall differences:** beta {_beta_test}; correlation {_rho_test}. A p-value above 0.05 does not reject equality at the 5% threshold, and does not establish equivalence."),
      mo.ui.table(pd.DataFrame(evidence['primary'])[['category','beta','legacy_beta','depth','levels','rho','athletes','matched','bouts']],selection=None,label='Category estimates and legacy beta comparison'),
      mo.accordion({'Beta pairwise differences, Holm adjustment':mo.ui.table(pd.DataFrame(evidence['beta']['pairs']),selection=None),
        'Correlation differences, separate Holm family':mo.ui.table(pd.DataFrame(evidence['rho']['pairs']),selection=None),
        'Age and participation sensitivities':mo.ui.table(pd.DataFrame(evidence['sensitivity_summary']),selection=None)}),
      mo.md('Intervals are 95% tournament-block bootstrap intervals with scores and eligibility refitted. Correlations compare score with official rank number, so stronger agreement is more negative. FIE agreement and unpredictability are distinct questions.')])
    return evidence,noise_panel


@app.cell
def _(evidence,mo,pd,plt,public_svg,zonal_panel):
    _primary=evidence['four_zone_primary']
    _regions=pd.DataFrame(_primary['category_means'])
    _fig,_ax=plt.subplots(figsize=(8,4))
    for _category,_rows in _regions.groupby('category',sort=False):
        _ax.plot(_rows['region'],_rows['mean'],marker='o',label=_category)
    _ax.axhline(0,color='gray');_ax.set_ylabel('Mean D: positive means better FIE placement');_ax.legend(ncol=3)
    differences_panel=mo.vstack([
      mo.md('## FIE differences\nD = (SpringRank rank − FIE rank)/(n − 1), using the common matched cohort within each category and average ties. **Positive D means better placement under FIE.** This is disagreement between systems, not proof of true ability or unfairness.'),
      public_svg(_fig,'Regional ranking discrepancies'),
      mo.md(f"**Primary classification: four championship zones.** Oceania shares the Asian championship pool and is included in Asia–Oceania. The equal-weight contrast is (Africa + Americas − Asia–Oceania − Europe)/2. Original-rank contrast with tournament uncertainty: `{_primary['original_rank_contrast']}`. Unresolved affiliations remain explicit; they are excluded from zone regressions."),
      mo.md('The hypothesis is that the zonal points slot improves FIE placement relative to SpringRank more for Africa/Americas. These are comparison groups for an exploratory hypothesis, not fixed intrinsic strength labels.'),
      mo.ui.table(_regions,selection=None,label='Four-zone category sample sizes and descriptive means'),
      zonal_panel,
      mo.accordion({'Adjusted four-zone associations':mo.ui.table(pd.DataFrame(_primary['adjusted']['D_original']['basic']['coefficients']),selection=None),
        'Participation and competition-mix adjustment':mo.ui.table(pd.DataFrame(_primary['adjusted']['D_original']['participation']['coefficients']),selection=None),
        'Four-zone sensitivity analyses':mo.ui.table(pd.DataFrame([{'analysis':r['analysis'],'n':r['n'],**r['contrasts']} for r in _primary['sensitivities']]),selection=None),
        'Secondary five-confederation descriptive results':mo.ui.table(pd.DataFrame(evidence['region_intervals']),selection=None),
        'Discrepancy prediction: country-grouped validation':mo.ui.table(pd.DataFrame(_primary['prediction']['discrepancy']['metrics']),selection=None),
        'Prediction errors by zone':mo.ui.table(pd.DataFrame(_primary['prediction']['discrepancy']['regional_errors']).replace({'region':{'Asia':'Asia-Oceania'}}),selection=None),
        'Later-bout prediction: chronological holdouts':mo.ui.table(pd.DataFrame(_primary['prediction']['bout'].get('metrics',[])),selection=None),
        'Chronological split and coverage audit':mo.ui.table(pd.DataFrame(_primary['prediction']['bout'].get('folds',[])),selection=None),
        'Calibration: training-only SpringRank':mo.ui.table(pd.DataFrame(_primary['prediction']['bout'].get('calibration',{}).get('springrank',[])),selection=None),
        'Calibration: biography-augmented model':mo.ui.table(pd.DataFrame(_primary['prediction']['bout'].get('calibration',{}).get('biography',[])),selection=None)}),
      mo.md(f"Later-bout coverage: {_primary['prediction']['bout'].get('coverage',0):.1%}. Event-bootstrap Brier improvement: `{_primary['prediction']['bout'].get('brier_improvement')}`. Invalid dates and current club/coach fields are excluded from historical prediction. Snapshot discrepancy prediction is not a forecast of future rankings."),
      mo.md('The logistic model also recalibrates the SpringRank coefficient, so any improvement cannot be attributed exclusively to biographies. Removing zonal bouts from SpringRank is a separate sensitivity from removing zonal ranking points. The geographic hypothesis remains exploratory and requires replication.')])
    return (differences_panel,)


@app.cell
def _(evidence,json,mo,pd,plt,public_svg):
    zonal_evidence=evidence['four_zone_primary']['zonal_points']
    _before=zonal_evidence['contrasts']['D_total_ties']
    _after=zonal_evidence['contrasts']['D_without_zonal']
    _change=zonal_evidence['contrasts']['D_change']
    _fraction=zonal_evidence['contrasts']['fraction_reduction']
    _regions=pd.DataFrame(zonal_evidence['regional_intervals'])
    _fig,_ax=plt.subplots(figsize=(8,4))
    for _metric,_offset,_label,_color in [('D_total_ties',-.12,'With zonal points','#287b8e'),('D_without_zonal',.12,'Without zonal points','#bb7045')]:
        _rows=_regions[_regions['metric'].eq(_metric)]
        _x=[i+_offset for i in range(len(_rows))]
        _ax.vlines(_x,_rows['low'],_rows['high'],color=_color)
        _ax.scatter(_x,_rows['estimate'],label=_label,color=_color)
    _ax.set_xticks(range(4),zonal_evidence['zone_labels'])
    _ax.axhline(0,color='gray',linewidth=.7);_ax.set_ylabel('Mean rank discrepancy D');_ax.legend();_fig.tight_layout()
    _correlations=[]
    for _row in zonal_evidence['correlations']:
        _correlations.append({'category':_row['category'],'n':_row['n'],
          'rho_with_zonal':_row['rho_total_ties']['estimate'],'rho_without_zonal':_row['rho_without_zonal']['estimate'],
          'change':_row['rho_change']['estimate'],'change_low':_row['rho_change']['low'],'change_high':_row['rho_change']['high'],
          'p_holm':_row['rho_change'].get('p_holm')})
    _adjusted=[{'target':target,'adjustment':kind,**model['contrast']} for target,models in zonal_evidence['adjusted'].items() for kind,model in models.items()]
    _valid=zonal_evidence['bootstrap']['valid_fraction']
    _p=f"{_change['p']:.4g}" if _change.get('p') is not None else 'suppressed: too few valid draws'
    _strength=evidence['four_zone_primary']['strength_conditioned']
    _points_model=next(r for r in _strength['models'] if r['outcome']=='zonal_points')
    zonal_panel=mo.vstack([
      mo.md('### What changes without zonal ranking points?\nAll **5,303 athlete/category totals reconcile exactly**: five best ordinary-event results, plus world/Olympic and zonal points. The cleaned headers identify every tournament. Zonal points occupy a separate slot, so deleting them does not promote a sixth ordinary result.'),
      public_svg(_fig,'Regional discrepancies with and without zonal ranking points'),
      mo.md(f"The equal-region contrast changes from **{_before['estimate']:.3f}** to **{_after['estimate']:.3f}**. Paired change: **{_change['estimate']:+.3f}**, 95% interval [{_change['low']:.3f}, {_change['high']:.3f}], p = {_p}. Relative contrast reduction: **{_fraction['estimate']:.1%}**. Valid paired tournament draws: {_valid:.1%} of 2,000 (seed 42)."),
      mo.md('Both orders use average point ties in the same matched cohort; the original official-rank result remains above. This isolates removing points from changing tie handling. It is a **mechanical points ablation**, with observed participation fixed. The percentage reduction is not a causal proportion explained, and the counterfactual is not an exact official FIE ordering: complete placement tie-breaks are unavailable.'),
      mo.ui.table(pd.DataFrame(_correlations),selection=None,label='Change in FIE agreement; paired intervals and six-test Holm adjustment'),
      mo.md('More negative score-versus-rank correlation means stronger agreement. Correlation intervals are unadjusted 95% paired bootstrap intervals; the six change tests use Holm-adjusted p-values.'),
      mo.md('#### Rewards at comparable strength\nThis complementary check uses SpringRank estimated **without zonal bouts**. It compares zonal points and normalized rank benefit among athletes with evidence of zonal participation, restricted to overlapping strength ranges in all four zones. Models adjust for category-specific cubic strength curves, category, age and handedness, with country-clustered uncertainty and Holm adjustment for two exploratory outcomes.'),
      mo.md(f"At comparable estimated strength, the equal-zone Africa/Americas versus Asia–Oceania/Europe contrast is **{_points_model['estimate']:+.1f} zonal points**, 95% country-clustered interval [{_points_model['low']:.1f}, {_points_model['high']:.1f}], across {_points_model['n']:,} athletes. This is a conditional association among evidenced participants, not a causal effect of entering zonals."),
      mo.ui.table(pd.DataFrame(_strength['models'])[['outcome','n','countries','estimate','low','high','p_holm']],selection=None,label='Conditional zonal-point and normalized rank-benefit contrasts'),
      mo.md(evidence['four_zone_primary']['strength_conditioned']['caveat']),
      mo.accordion({'Every-row points reconciliation':mo.ui.table(pd.DataFrame(zonal_evidence['reconciliation']),selection=None),
        'Cleaned event dictionary':mo.ui.table(pd.DataFrame(zonal_evidence['events']),selection=None),
        'Adjusted contrasts before, after and for the paired change':mo.ui.table(pd.DataFrame(_adjusted),selection=None),
        'Regional paired changes':mo.ui.table(_regions,selection=None),
        'Regional contrasts within each category':mo.ui.table(pd.DataFrame(zonal_evidence['category_regional_contrasts']),selection=None),
        'Strength comparison coverage':mo.ui.table(pd.DataFrame(evidence['four_zone_primary']['strength_conditioned']['coverage']),selection=None),
        'Common-strength support and exclusions':mo.ui.table(pd.DataFrame(evidence['four_zone_primary']['strength_conditioned']['support_audit']),selection=None),
        'Sensitivity excluding sparse retained zone/category cells':mo.ui.table(pd.DataFrame(_strength['sensitivity_models']),selection=None),
        'Fixed FIE top-200 cohort sensitivity':mo.ui.table([zonal_evidence['top200_sensitivity']['contrasts']],selection=None),
        'Remove both zonal points and zonal bouts':mo.ui.table([zonal_evidence['both_zonal_points_and_bouts_removed']['contrasts']],selection=None)}),
      mo.md(f"[FIE counting and tie rules, o.108]({zonal_evidence['rule_source']}). Tournament resampling captures bout-network uncertainty conditional on the archived points snapshot; it does not represent uncertainty across independent FIE seasons."),
      mo.download(json.dumps(zonal_evidence,ensure_ascii=False).encode('utf-8'),filename='zonal-points-analysis.json',mimetype='application/json',label='Download zonal-points analysis')])
    return zonal_evidence,zonal_panel


@app.cell
def _(mo,snapshot):
    fantasy_category_a=mo.ui.dropdown(options={v:k for k,v in snapshot['category_labels'].items()},value="Women's foil",label='Fencer A category',allow_select_none=False)
    fantasy_category_b=mo.ui.dropdown(options={v:k for k,v in snapshot['category_labels'].items()},value="Women's foil",label='Fencer B category',allow_select_none=False)
    return fantasy_category_a,fantasy_category_b


@app.cell
def _(fantasy_category_a,fantasy_category_b,mo,snapshot):
    fantasy_league_a=next(v for v in snapshot['variants'] if v['population']=='Senior events' and v['minimum_bouts']==5 and v['category']==fantasy_category_a.value)
    fantasy_league_b=next(v for v in snapshot['variants'] if v['population']=='Senior events' and v['minimum_bouts']==5 and v['category']==fantasy_category_b.value)
    _options_a={snapshot['athletes'][r[0]]['name']+' ('+snapshot['athletes'][r[0]]['country']+')':r[0] for r in sorted(fantasy_league_a['rows'],key=lambda r:r[1],reverse=True)}
    _options_b={snapshot['athletes'][r[0]]['name']+' ('+snapshot['athletes'][r[0]]['country']+')':r[0] for r in sorted(fantasy_league_b['rows'],key=lambda r:r[1],reverse=True)}
    fantasy_athlete_a=mo.ui.dropdown(options=_options_a,value=list(_options_a)[0],label='Fencer A',allow_select_none=False)
    fantasy_athlete_b=mo.ui.dropdown(options=_options_b,value=list(_options_b)[1],label='Fencer B',allow_select_none=False)
    return fantasy_league_a,fantasy_league_b,fantasy_athlete_a,fantasy_athlete_b


@app.cell
def _(fantasy_athlete_a,fantasy_athlete_b,fantasy_category_a,fantasy_category_b,fantasy_chance,fantasy_league_a,fantasy_league_b,mo):
    _a=next(r for r in fantasy_league_a['rows'] if r[0]==fantasy_athlete_a.value)
    _b=next(r for r in fantasy_league_b['rows'] if r[0]==fantasy_athlete_b.value)
    _ma=sum(r[1] for r in fantasy_league_a['rows'])/len(fantasy_league_a['rows'])
    _mb=sum(r[1] for r in fantasy_league_b['rows'])/len(fantasy_league_b['rows'])
    fantasy_probability=fantasy_chance(_a[1],fantasy_league_a['beta'],_b[1],fantasy_league_b['beta'],_ma,_mb)
    _cross=fantasy_category_a.value!=fantasy_category_b.value
    fantasy_panel=mo.vstack([mo.md('## Fantasy matchups\nSenior events, minimum five bouts.'),
      mo.hstack([fantasy_category_a,fantasy_category_b],wrap=True),mo.hstack([fantasy_athlete_a,fantasy_athlete_b],wrap=True),
      mo.md(('### Fantasy matchup—hypothetical comparison' if _cross else '### Same-category model probability')+f'\n\n**Fencer A: {fantasy_probability:.1%} · Fencer B: {1-fantasy_probability:.1%}**'),
      mo.md('Same-category: logistic(2β(sA − sB)). Cross-category: z = 2β(s − category mean), then logistic(zA − zB). **Aligning category means is an entertainment convention: disconnected categories have no empirically established common ability scale, even after beta scaling.** These are fun comparisons, not validated cross-category bout predictions.')])
    return fantasy_panel,fantasy_probability


@app.cell(hide_code=True)
def _(mo,noise_panel,differences_panel,fantasy_panel,rank_controls,rank_view_6,rank_view_8,rank_view_9,rank_view_10,rank_view_12,rank_view_13,rank_view_14):
    mo.ui.tabs({'Noise across categories':noise_panel,'FIE differences':differences_panel,'Athlete rankings':mo.vstack([rank_controls,rank_view_6,rank_view_8,rank_view_9,rank_view_10,rank_view_12,rank_view_13,rank_view_14]),'Fantasy matchups':fantasy_panel})
    return

if __name__ == '__main__':
    app.run()
