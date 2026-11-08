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
    if snapshot.get('schema_version') != 1 or snapshot.get('data_kind') != 'actual_research_outputs':
        raise ValueError('Expected a supported snapshot of actual research outputs.')
    return (snapshot,)


@app.cell(hide_code=True)
def _(mo, snapshot):
    mo.md(f"""
    # Fencing network rankings

    Explore rankings inferred from **actual 2025/26 fencing bouts** using SpringRank.
    Compare the ordering with archived FIE standings and inspect how the results change
    with the athlete population and minimum participation threshold.

    This is a saved analysis snapshot, generated **{snapshot['generated_at_utc'][:10]}**.
    The controls select independently calculated results; data collection stays in the local research notebook.
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
    mo.vstack([mo.hstack([population, category, minimum_bouts], justify='start', wrap=True), elite_only])
    return athlete_search, category, elite_only, minimum_bouts, population, top_count


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
    mo.md(f"""
    **{len(rankings):,} athletes** · **{selected_variant['retained_bouts']:,} retained bouts** ·
    **{_matched_count:,} matched to archived FIE ranks** ({_matched_count / len(rankings):.1%} coverage)

    Higher SpringRank scores indicate stronger observed performance. Scores are centered separately
    within each fitted category and population; they do not support cross-category match predictions.
    """)
    return


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
    mo.md(f"""
    ## Agreement with FIE standings

    **Spearman correlation: {_stat}**, using **{len(comparison):,} athletes** in {_scope}.
    A negative correlation is expected: the strongest model scores should correspond to the
    smallest official rank numbers. Changing the elite filter changes the comparison sample.
    """)
    return


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
        _svg = _buffer.getvalue().replace('<svg ', '<svg role="img" aria-label="SpringRank score versus archived FIE rank" ', 1)
        _plot = mo.Html(_svg)
        plt.close(_fig)
    else:
        _plot = mo.md('No FIE comparison is available for this selection.')
    _plot
    return


@app.cell(hide_code=True)
def _(athlete_search, mo, top_count):
    mo.vstack([mo.md('## Explore the calculated rankings'), mo.hstack([athlete_search, top_count], justify='start', wrap=True)])
    return


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
        _svg = _buffer.getvalue().replace('<svg ', '<svg role="img" aria-label="Highest calculated scores in the current athlete search" ', 1)
        _chart = mo.Html(_svg)
        plt.close(_fig)
    else:
        _chart = mo.md('No athletes match that search.')
    _chart
    return


@app.cell
def _(category, minimum_bouts, mo, population, visible_rankings):
    _table = visible_rankings[['model_rank', 'athlete', 'country', 'score', 'retained_bouts', 'fie_rank']].rename(columns={
        'model_rank': 'Model rank', 'athlete': 'Athlete', 'country': 'Country', 'score': 'SpringRank score',
        'retained_bouts': 'Retained bouts', 'fie_rank': 'Archived FIE rank'})
    mo.vstack([
        mo.ui.table(_table, pagination=True, page_size=15, selection=None, show_download=False, label='Calculated athlete rankings'),
        mo.download(data=_table.to_csv(index=False).encode('utf-8'), filename=f"{category.value}-{population.value.replace(' ', '-')}-min{minimum_bouts.value}.csv", mimetype='text/csv', label='Download these rankings'),
    ])
    return


@app.cell(hide_code=True)
def _(mo, snapshot):
    _audit = snapshot['audit']
    mo.accordion({
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
        are not a common scale of ability. No cross-weapon fantasy probabilities are presented here.
        """),
        'Sources and reproducibility': mo.md("""
        Analysis: [Oly-Fencing-Networks](https://github.com/Zargon42/Oly-Fencing-Networks).
        Bout dataset credited to Andrew Fischl (CyrusOfChaos). SpringRank utilities were adapted from
        Caterina De Bacco's [Probabilistic Inference in Networks material](https://github.com/cdebacco/ds_prob_inf_net).
        Official ranking source: [FIE](https://fie.org/athletes).

        The site contains a compact, actual-results snapshot and its input hashes. It does not contain
        raw event files or biography caches. The repository's `scripts/export_public_snapshot.py` rebuilds
        the snapshot locally from the source data. Browser controls explore the saved fits without fetching
        FIE profiles. Internet access is needed initially for the Python browser runtime and packages.
        """),
    })
    return


if __name__ == '__main__':
    app.run()
