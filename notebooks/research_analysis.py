"""Research notebook: noise, FIE discrepancies, and biographical prediction."""

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    app_title="Fencing research: noise and ranking discrepancies",
)

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fencing networks: noise and ranking discrepancies

    **Questions.** Do the six weapon/gender categories differ in model-estimated noise?
    How closely do their rankings agree with FIE standings? Does the zonal-points slot
    help explain geographic disagreement, and can biographies improve prediction?

    This notebook is the executable research record. Read and run it from top to bottom.
    Bootstrap stages resume matching local checkpoints; models run in the notebook.
    The final section is the **only** writer of the report and public snapshot.

    **Contents:** 1. Inputs and cohort · 2. Noise · 3. Four-zone discrepancy ·
    4. Zonal rewards at comparable strength · 5. Prediction · 6. Sensitivities ·
    7. Validation and interpretation · 8. Export.
    """)
    return


@app.cell
def settings():
    project_root = Path(mo.notebook_location()).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    settings = {
        "population": "Senior events", "minimum_bouts": 5,
        "replicates": 2000, "seed": 42, "workers": 4,
        "strength_support": (0.05, 0.95), "strength_degree": 3,
    }
    assert settings["population"] == "Senior events" and settings["minimum_bouts"] == 5, "The primary bootstrap protocol is senior/minimum five; use section 6 for other cohorts."
    return project_root, settings


@app.cell
def methods(project_root):
    assert (project_root / "src/research.py").exists(), "Open this notebook from the project checkout."
    from src import research, ranking_points
    from src import notebook_data as data_tools
    from src import zonal_strength, research_publication as publication

    return data_tools, publication, ranking_points, research, zonal_strength


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1. Inputs and the analysis cohort

    The primary population is **senior events, at least five recorded bouts**, with
    each weapon/gender category fitted separately. Eligibility is applied before
    opponent filtering and selection of the largest connected component.

    The local inputs are pool/DE bouts, six archived detailed FIE rankings, and the
    enriched FIE biography file described in `DATA.md`. Original inputs are preserved.
    Only derived results are exported to GitHub Pages.

    **Identity matching:** FIE links require a unique normalized name in the full
    category input, before participation filtering. Ambiguous spellings are excluded
    consistently at all thresholds, even if only one spelling survives a stricter filter.
    """)
    return


@app.cell
def load_bouts(project_root, research):
    pools = pd.read_csv(project_root / "CoC dataset/Pools.csv", low_memory=False)
    direct_elimination = pd.read_csv(project_root / "CoC dataset/DEs.csv", low_memory=False)
    raw_bouts = pd.concat([pools.assign(Stage="Pool"), direct_elimination.assign(Stage="DE")], ignore_index=True)
    bouts = research.prepare_bouts(raw_bouts)
    input_audit = pd.DataFrame([{
        "Recorded rows": len(raw_bouts), "Valid fenced outcomes": len(bouts),
        "Invalid dates": int(bouts["Date"].isna().sum()),
        "First valid date": bouts["Date"].min().date(), "Last valid date": bouts["Date"].max().date(),
    }])
    mo.ui.table(input_audit, selection=None)
    return bouts, raw_bouts


@app.cell
def reconcile_points(data_tools, project_root):
    point_tables, event_dictionary, point_audit = data_tools.reconcile_rankings(project_root / "rankings")
    assert point_audit["unreconciled"].sum() == 0
    mo.vstack([
        mo.ui.table(point_audit.drop(columns="input_sha256"), selection=None, label="Every ranking total reconciled"),
        mo.accordion({"Original and cleaned tournament headers": mo.ui.table(event_dictionary, selection=None)}),
    ])
    return event_dictionary, point_audit, point_tables


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Reconciliation rule.** Five best World Cup/Grand Prix/Satellite results, plus
    separate world/Olympic and zonal slots, reproduce the supplied totals. Point values
    already include multipliers. Rotated headers are repaired and original headers
    retained in the dictionary. Event years follow the archived 2025/26 season.
    [FIE Organisation Rules, o.108](https://static.fie.org/uploads/38/190670-Organisation%20rules%20ang.pdf).

    **Geographic level.** The primary comparison uses four championship zones:
    Africa, Americas, **Asia–Oceania**, and Europe. Confederation membership remains a
    separate attribute; the five-confederation view appears only as a sensitivity.
    Unknown affiliations are retained explicitly and excluded from zone regressions.
    """)
    return


@app.cell
def classify_and_fit(bouts, data_tools, project_root, research, settings):
    federation_source = data_tools.federation_lookup(project_root / "data/research/fie_regions.json")
    federations = federation_source["federations"]
    network_leagues, tournament_blocks = research.encode_leagues(
        bouts, project_root / "rankings", federations, population=settings["population"])
    primary_fits = {code: research.fit_league(league, settings["minimum_bouts"])
                    for code, league in network_leagues.items()}
    zone_leagues = data_tools.championship_leagues(network_leagues)
    zone_federations = data_tools.championship_federations(federations)
    zone_fits = data_tools.championship_fits(primary_fits)
    primary_summary = pd.DataFrame([summary for _, summary in primary_fits.values()])
    mo.ui.table(primary_summary[["category", "athletes", "bouts", "matched", "components_before_filter"]],
                selection=None, label="Primary fitted cohorts and FIE matching")
    return (
        federations,
        network_leagues,
        primary_fits,
        primary_summary,
        tournament_blocks,
        zone_federations,
        zone_fits,
        zone_leagues,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Noise across the six categories

    SpringRank scores use a common fitting convention. We estimate beta from a
    conditional likelihood that counts **each unordered opponent pair once**. The
    legacy sparse implementation gave reciprocal pairs extra weight; its beta is
    retained for comparison. Scores are not independently standardized.

    $$P(i>j)=\operatorname{logistic}\{2eta(s_i-s_j)\}.$$

    Higher beta gives a sharper probability curve at a common score gap. Depth and
    hierarchy levels describe the fitted score distribution. Neither beta nor agreement
    with FIE is a direct measurement of intrinsic luck or true ability.
    """)
    return


@app.cell
def noise_estimates(primary_summary):
    mo.ui.table(primary_summary[["category", "beta", "legacy_beta", "depth", "levels", "iqr_levels", "rho"]],
                selection=None, label="Corrected estimates, hierarchy and FIE agreement")
    return


@app.cell
def noise_bootstrap(
    network_leagues,
    primary_summary,
    project_root,
    research,
    settings,
    tournament_blocks,
):
    with mo.status.progress_bar(total=settings["replicates"], title="Tournament bootstrap: six-category network refits") as _progress:
        noise_draws = research.run_bootstrap(
            network_leagues, tournament_blocks, replicates=settings["replicates"], seed=settings["seed"],
            workers=settings["workers"], checkpoint=project_root / "data/research/tournament_bootstrap.json",
            progress=lambda done, total: _progress.update(increment=1))
    beta_comparison = research.bootstrap_tables(primary_summary["beta"], noise_draws["beta"])
    rho_comparison = research.bootstrap_tables(primary_summary["rho"], noise_draws["rho"])
    mo.md(f"Completed **{len(noise_draws['beta']):,} draws**, seed {settings['seed']}. Joint valid: "
          f"beta **{beta_comparison['valid_fraction']:.1%}**; correlation **{rho_comparison['valid_fraction']:.1%}**.")
    return beta_comparison, noise_draws, rho_comparison


@app.cell
def noise_intervals(beta_comparison, data_tools, rho_comparison):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _ax, _result, _title in [(_axes[0], beta_comparison, "Beta"), (_axes[1], rho_comparison, "FIE agreement: Spearman rho")]:
        _table = pd.DataFrame(_result["intervals"])
        _ax.vlines(_table["category"], _table["low"], _table["high"], color="#287b8e")
        _ax.scatter(_table["category"], _table["estimate"], color="#287b8e")
        _ax.set_title(_title)
    _fig.tight_layout()
    _svg = data_tools.svg(_fig)
    plt.close(_fig)
    mo.Html(_svg)
    return


@app.cell
def probability_curves(data_tools, primary_summary):
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _gap = np.linspace(0, 1, 100)
    for _row in primary_summary.to_dict("records"):
        _ax.plot(_gap, 1 / (1 + np.exp(-2 * _row["beta"] * _gap)), label=_row["category"])
    _ax.set(xlabel="Common SpringRank score gap", ylabel="Model-implied favorite win probability")
    _ax.legend(ncol=3)
    _svg = data_tools.svg(_fig)
    plt.close(_fig)
    mo.Html(_svg)
    return


@app.cell
def noise_tests(beta_comparison, rho_comparison):
    _tests = pd.DataFrame([
        {"metric": "beta", **(beta_comparison.get("omnibus") or {"p": None})},
        {"metric": "Spearman rho", **(rho_comparison.get("omnibus") or {"p": None})},
    ])
    mo.vstack([
        mo.ui.table(_tests, selection=None, label="Overall equality tests"),
        mo.accordion({
            "15 beta differences, Holm correction": mo.ui.table(pd.DataFrame(beta_comparison["pairs"]), selection=None),
            "15 correlation differences, separate Holm family": mo.ui.table(pd.DataFrame(rho_comparison["pairs"]), selection=None),
        }),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Inference.** Shared tournament blocks are resampled jointly across categories;
    eligibility, connected components, scores and FIE matching are refitted. Intervals
    are 95% percentile bootstrap intervals. Tests use centered bootstrap contrasts,
    with separate Holm families for beta and correlation differences. Formal inference
    requires at least 90% joint valid draws. Bootstrap draws are not independent
    observations for ANOVA. A non-significant result does not establish equivalence.

    Spearman rho compares scores with official rank numbers, so stronger agreement is
    more negative. This measures concurrent agreement, not future-bout accuracy.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Four-zone ranking discrepancy and zonal points

    Within each category's **common matched cohort**, both ranks use 1 as best and
    average ties. Define $D=(r_{\mathrm{SR}}-r_{\mathrm{FIE}})/(n-1)$: positive D means
    FIE places an athlete better than SpringRank does.

    **Hypothesis:** the zonal slot benefits FIE placement relative to SpringRank more
    for Africa/Americas than Asia–Oceania/Europe. The equal-zone contrast is
    $(ar D_A+ar D_M-ar D_{AO}-ar D_E)/2$.

    The counterfactual removes zonal points with observed participation held fixed.
    It does not promote a sixth ordinary result. Both point orders use average ties,
    so tie-rule changes are not confused with deleting points. The original official
    ordinal ranks remain a sensitivity; exact placement tie-breaks are not claimed.
    """)
    return


@app.cell
def zone_discrepancies(data_tools, point_tables, zone_fits):
    point_frames = data_tools.points_frames(zone_fits, point_tables)
    point_data = pd.concat(point_frames.values(), ignore_index=True)
    zone_means = data_tools.category_zone_means(point_frames)
    mo.ui.table(zone_means, selection=None, label="Four-zone matched samples and discrepancies")
    return point_data, point_frames


@app.cell
def zone_bootstrap(
    data_tools,
    point_frames,
    point_tables,
    project_root,
    ranking_points,
    settings,
    tournament_blocks,
    zone_leagues,
):
    with mo.status.progress_bar(total=settings["replicates"], title="Four-zone paired tournament resampling") as _progress:
        zone_draws = ranking_points.paired_bootstrap(
            zone_leagues, point_tables, tournament_blocks, replicates=settings["replicates"], seed=settings["seed"],
            workers=settings["workers"], checkpoint=project_root / "data/research/four_zone_points_bootstrap.json",
            progress=lambda done, total: _progress.update(increment=1))
    zone_inference = data_tools.points_inference(point_frames, zone_draws)
    mo.ui.table(pd.DataFrame([{"metric": key, **value} for key, value in zone_inference["contrasts"].items()]),
                selection=None, label="Paired zone contrasts and changes")
    return zone_draws, zone_inference


@app.cell
def zone_plot(data_tools, zone_inference):
    _intervals = pd.DataFrame(zone_inference["regional_intervals"])
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _metric, _offset, _label, _color in [("D_total_ties", -.12, "With zonal points", "#287b8e"),
                                           ("D_without_zonal", .12, "Without zonal points", "#bb7045")]:
        _rows = _intervals[_intervals["metric"].eq(_metric)]
        _x = np.arange(4) + _offset
        _ax.vlines(_x, _rows["low"], _rows["high"], color=_color)
        _ax.scatter(_x, _rows["estimate"], color=_color, label=_label)
    _ax.set_xticks(range(4), [data_tools.ZONE_LABELS[z] for z in data_tools.ZONE_CODES])
    _ax.axhline(0, color="gray", linewidth=.7)
    _ax.set_ylabel("Mean discrepancy D"); _ax.legend(); _fig.tight_layout()
    _svg = data_tools.svg(_fig)
    plt.close(_fig)
    mo.Html(_svg)
    return


@app.cell
def biographies_and_features(
    bouts,
    project_root,
    research,
    zone_fits,
    zone_leagues,
):
    biographies, biography_coverage = research.biography_features(
        project_root / "data/fie_biographies/athletes_enriched.csv", zone_leagues, bouts["Date"].max())
    features = research.discrepancy_features(zone_leagues, zone_fits, biographies)
    trait_coverage = pd.DataFrame([{
        "category": code, "athletes": len(frame), "age_known": int(frame["age"].notna().sum()),
        "hand_known": int(frame["hand"].ne("Unknown").sum()), "height_known": int(frame["height"].notna().sum()),
    } for code, frame in biographies.items()])
    mo.accordion({"Biography coverage and missingness": mo.ui.table(trait_coverage, selection=None)})
    return biographies, biography_coverage, features


@app.cell
def adjusted_zone_models(data_tools, features, point_data):
    zone_adjustments = data_tools.adjusted_contrasts(features, point_data)
    adjusted_summary = pd.DataFrame([
        {"outcome": target, "adjustment": kind, **model["contrast"]}
        for target, models in zone_adjustments.items() for kind, model in models.items()
    ])
    mo.ui.table(adjusted_summary, selection=None, label="Country-clustered zone contrasts")
    return (zone_adjustments,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    The first adjusted model includes category, age and handedness; the second adds
    participation and competition mix. These are country-clustered conditional
    associations. The paired tournament intervals propagate network uncertainty
    conditional on the fixed archived points snapshot; they do not measure variation
    across independent seasons. A large mechanical reduction does not establish true
    ability, unfairness or a causal effect of entering zonals.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 4. Zonal rewards at comparable estimated strength

    This complementary test estimates SpringRank **without zonal bouts**, then uses
    within-category strength percentiles. Evidence of participation is a recorded CHZ
    score (including explicit zero) or an observed zonal bout. Missing both does not
    prove nonparticipation.

    To limit extrapolation, each category is restricted to the intersection of the
    four zones' 5th-to-95th percentile strength ranges. Models compare zonal points
    and normalized rank benefit, using category-specific cubic strength curves,
    category, age and hand. Uncertainty is country-clustered; the two exploratory
    outcomes receive Holm correction. Strength-estimation uncertainty is not fully
    propagated by this regression.
    """)
    return


@app.cell
def nonzonal_strength(
    bouts,
    features,
    federations,
    point_data,
    point_tables,
    project_root,
    research,
    settings,
    zonal_strength,
):
    nonzonal_leagues, _ = research.encode_leagues(
        bouts, project_root / "rankings", federations, population=settings["population"], exclude_zonal=True)
    nonzonal_fits = {code: research.fit_league(league, settings["minimum_bouts"])
                     for code, league in nonzonal_leagues.items()}
    strength_cohort, strength_coverage, support_audit, strength_descriptions = zonal_strength.prepare_strength_cohort(
        point_data, nonzonal_fits, features, point_tables, *settings["strength_support"])
    mo.vstack([
        mo.ui.table(support_audit, selection=None, label="Common strength support and exclusions"),
        mo.accordion({"Participation evidence and coverage": mo.ui.table(strength_coverage, selection=None)}),
    ])
    return (
        nonzonal_fits,
        strength_cohort,
        strength_coverage,
        strength_descriptions,
        support_audit,
    )


@app.cell
def strength_models(research, settings, strength_cohort, zonal_strength):
    strength_models = [zonal_strength.conditioned_model(strength_cohort, target, degree=settings["strength_degree"])
                       for target in ["zonal_points", "rank_benefit"]]
    for _row, _p in zip(strength_models, research.holm([r["p"] for r in strength_models])):
        _row["p_holm"] = float(_p)
    dense_cohort, dense_categories = zonal_strength.dense_support(strength_cohort, minimum=5)
    strength_sensitivity = [zonal_strength.conditioned_model(dense_cohort, target, degree=settings["strength_degree"])
                           for target in ["zonal_points", "rank_benefit"]]
    mo.vstack([
        mo.ui.table(pd.DataFrame(strength_models).drop(columns="conditioning"), selection=None, label="Rewards at comparable non-zonal strength"),
        mo.accordion({"Exclude categories with sparse retained zone cells": mo.ui.table(pd.DataFrame(strength_sensitivity), selection=None)}),
    ])
    return dense_categories, strength_models, strength_sensitivity


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 5. Two separate prediction tasks

    **A. Snapshot discrepancy:** zero-discrepancy baseline, ridge and gradient boosting.
    Outer validation and inner tuning are grouped by country. Ambiguous cross-country
    athlete names are conservatively excluded. Report MAE, out-of-sample R-squared
    and errors by zone. This is not a forecast of future ranking discrepancies.

    **B. Later bouts:** rolling whole-event holdouts. Scores, beta, imputation and models
    use training data only. Compare SpringRank probabilities with a logistic model
    adding biography differences. Predictions must complement when athlete order is
    swapped. Invalid dates are excluded; current club/coach fields are not historical
    predictors. Age and handedness are primary; height and club coverage are sparse.
    """)
    return


@app.cell
def discrepancy_prediction(data_tools, features, research, settings):
    with mo.status.progress_bar(total=1, title="Country-grouped validation and inner tuning") as _progress:
        discrepancy_prediction = research.discrepancy_prediction(features, seed=settings["seed"])
        _progress.update()
    mo.vstack([
        mo.ui.table(pd.DataFrame(discrepancy_prediction["metrics"]), selection=None, label="Held-out-country performance"),
        mo.accordion({"Errors by zone": mo.ui.table(pd.DataFrame(discrepancy_prediction["regional_errors"]).replace({"region": data_tools.ZONE_LABELS}), selection=None),
                      "Country and athlete separation audit": mo.ui.table(pd.DataFrame(discrepancy_prediction["folds"]), selection=None)}),
    ])
    return (discrepancy_prediction,)


@app.cell
def bout_prediction(
    biographies,
    research,
    settings,
    zone_federations,
    zone_leagues,
):
    with mo.status.progress_bar(total=18, title="Training-only models and chronological event holdouts") as _progress:
        bout_prediction = research.rolling_bout_prediction(
            zone_leagues, biographies, zone_federations, seed=settings["seed"],
            progress=lambda done, total: _progress.update(increment=1))
    mo.vstack([
        mo.ui.table(pd.DataFrame(bout_prediction["metrics"]).query("category == 'All'"), selection=None, label="Later-bout performance"),
        mo.md(f"Covered **{bout_prediction['coverage']:.1%}** of test bouts. Paired Brier improvement: `{bout_prediction['brier_improvement']}`."),
        mo.accordion({"Chronology, coverage and swap checks": mo.ui.table(pd.DataFrame(bout_prediction["folds"]), selection=None),
                      "Category metrics": mo.ui.table(pd.DataFrame(bout_prediction["metrics"]), selection=None)}),
    ])
    return (bout_prediction,)


@app.cell
def calibration(bout_prediction, data_tools):
    _fig, _ax = plt.subplots(figsize=(5, 4))
    for _model, _rows in bout_prediction["calibration"].items():
        _table = pd.DataFrame(_rows)
        if not _table.empty:
            _ax.plot(_table["mean_probability"], _table["observed_win_fraction"], marker="o", label=_model)
    _ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    _ax.set(xlabel="Mean predicted probability", ylabel="Observed win fraction")
    _ax.legend(); _fig.tight_layout()
    _svg = data_tools.svg(_fig)
    plt.close(_fig)
    mo.Html(_svg)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    The biography logistic model can also recalibrate the SpringRank coefficient.
    Any predictive difference cannot be attributed exclusively to biographies.
    Calibration, coverage and held-out scores should be read together; a small
    apparent improvement without a clear interval is not evidence of useful lift.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 6. Sensitivity analyses

    Repeat the score fits for senior-only and combined ages, with minimum participation
    of one, five and ten bouts. For the zone question, also examine the fixed FIE top-200
    cohort and removal of zonal **bouts** from SpringRank. Removing bouts changes the
    fitted cohort; it is distinct from removing points with the cohort fixed.

    The five-confederation view retains Oceania separately as a descriptive sensitivity.
    It does not define a fifth independent championship points pool.
    """)
    return


@app.cell
def participation_sensitivities(
    bouts,
    federations,
    network_leagues,
    primary_fits,
    project_root,
    research,
):
    variant_leagues = {"Senior events": network_leagues}
    variant_leagues["All recorded ages"], _ = research.encode_leagues(bouts, project_root / "rankings", federations, population="All recorded ages")
    variant_fits = {}
    for _population, _leagues in variant_leagues.items():
        for _minimum in [1, 5, 10]:
            for _code, _league in _leagues.items():
                variant_fits[(_population, _minimum, _code)] = (primary_fits[_code] if _population == "Senior events" and _minimum == 5
                                                              else research.fit_league(_league, _minimum))
    sensitivity_summary = pd.DataFrame([{"population": p, "minimum": m, **summary}
                                        for (p, m, c), (frame, summary) in variant_fits.items()])
    mo.ui.table(sensitivity_summary[["population", "minimum", "category", "beta", "rho", "athletes", "matched"]], selection=None)
    return variant_fits, variant_leagues


@app.cell
def zone_sensitivities(
    data_tools,
    noise_draws,
    nonzonal_fits,
    point_tables,
    primary_fits,
    ranking_points,
    zone_fits,
):
    zone_sensitivities = []
    for _label, _fits, _elite in [("Primary", zone_fits, False),
                                 ("Exclude zonal bouts", data_tools.championship_fits(nonzonal_fits), False),
                                 ("FIE top 200", zone_fits, True)]:
        _frames = data_tools.points_frames(_fits, point_tables, elite=_elite)
        _contrasts, _correlations = ranking_points.summarize(list(_frames.values()))
        zone_sensitivities.append({"analysis": _label, "n": sum(len(f) for f in _frames.values()),
                                   "contrasts": _contrasts, "correlations": _correlations})
    confederation_summary = data_tools.confederation_intervals(primary_fits, noise_draws)
    mo.vstack([
        mo.ui.table(pd.DataFrame([{"analysis": r["analysis"], "n": r["n"], **r["contrasts"]} for r in zone_sensitivities]), selection=None),
        mo.accordion({"Secondary five-confederation description": mo.ui.table(pd.DataFrame(confederation_summary), selection=None)}),
    ])
    return confederation_summary, zone_sensitivities


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 7. Validation and interpretation

    The checks below cover likelihood boundaries, discrepancy signs and ties, unchanged
    cohorts, zone membership, seeded resampling and fantasy-probability symmetry.
    Prediction functions also assert country/athlete separation, event chronology and
    complementary swapped predictions. Full results remain conditional on the observed
    population, recorded events and archived FIE snapshot.
    """)
    return


@app.cell
def validation(
    features,
    point_audit,
    point_data,
    primary_fits,
    research,
    zone_federations,
):
    assert point_audit["unreconciled"].sum() == 0
    assert len(point_data) == len(features)
    assert not point_data["region"].eq("Oceania").any()
    _matched = pd.concat([f.dropna(subset=["D"]) for f, s in primary_fits.values()])
    assert point_data["confederation_region"].eq("Oceania").sum() == _matched["region"].eq("Oceania").sum()
    assert np.allclose(research.discrepancy([3, 3, 1], [1, 1, 3]), 0)
    assert research.discrepancy([1, 2, 3], [1, 2, 3])[0] > 0
    assert research.beta_from_pairs(np.zeros(2), np.ones(2), np.ones(2))[1] == "unidentified"
    assert research.beta_from_pairs(np.ones(2), np.ones(2), np.ones(2))[1] == "separated_infinite"
    assert research.beta_from_pairs(np.ones(2), np.zeros(2), np.ones(2))[1] == "boundary_zero"
    assert research.fantasy_probability(.3, 1.4, .3, 1.4) == .5
    assert abs(research.fantasy_probability(.3, 1.4, -.2, 1.8) + research.fantasy_probability(-.2, 1.8, .3, 1.4) - 1) < 1e-12
    _delta = np.array([.5, 1., .5])
    _wins, _totals = np.array([8., 7., 6.]), np.array([10., 7., 10.])
    _beta, _status = research.beta_from_pairs(_delta, _wins, _totals)
    _legacy, _ = research.beta_from_pairs(_delta, _wins, _totals, legacy=True)
    assert _status == "finite" and abs(_beta - _legacy) > .01
    assert abs(np.dot(_delta, _wins - _totals / (1 + np.exp(-2 * _beta * _delta)))) < 1e-8
    assert zone_federations.get("AIN", {}).get("region", "Unknown") == "Unknown"
    checks = {"all_totals_reconciled": True, "cohort_preserved": True, "four_zones_only": True,
              "ties_and_signs": True, "beta_boundaries": True, "corrected_likelihood": True, "unknown_affiliations": True, "matchup_symmetry": True}
    mo.md("**Checks passed:** exact totals, matched cohorts, four-zone membership, rank signs/ties, beta boundaries and matchup symmetry.")
    return (checks,)


@app.cell
def seed_reproducibility(
    network_leagues,
    noise_draws,
    point_tables,
    ranking_points,
    research,
    settings,
    tournament_blocks,
    zone_draws,
    zone_leagues,
):
    _repeat_noise = research.run_bootstrap(network_leagues, tournament_blocks, replicates=3, seed=settings["seed"], workers=1)
    _repeat_zones = ranking_points.paired_bootstrap(zone_leagues, point_tables, tournament_blocks, replicates=3, seed=settings["seed"], workers=1)
    assert np.allclose(_repeat_noise["beta"], noise_draws["beta"][:3], equal_nan=True)
    assert np.allclose(_repeat_noise["rho"], noise_draws["rho"][:3], equal_nan=True)
    assert _repeat_zones["results"] == zone_draws["results"][:3]
    reproducibility_passed = True
    mo.md("**Seed check passed:** three fresh serial draws reproduce each bootstrap's parallel draws.")
    return (reproducibility_passed,)


@app.cell
def assemble_results(
    beta_comparison,
    biography_coverage,
    bout_prediction,
    bouts,
    checks,
    confederation_summary,
    data_tools,
    dense_categories,
    discrepancy_prediction,
    event_dictionary,
    noise_draws,
    point_audit,
    point_frames,
    primary_summary,
    project_root,
    publication,
    raw_bouts,
    reproducibility_passed,
    rho_comparison,
    strength_coverage,
    strength_descriptions,
    strength_models,
    strength_sensitivity,
    support_audit,
    variant_fits,
    variant_leagues,
    zone_adjustments,
    zone_federations,
    zone_inference,
    zone_sensitivities,
):
    strength_results = {
        "models": strength_models, "sensitivity_models": strength_sensitivity, "dense_categories": dense_categories,
        "coverage": strength_coverage.replace({"region": data_tools.ZONE_LABELS}).to_dict("records"),
        "support_audit": support_audit.to_dict("records"),
        "descriptives": strength_descriptions.replace({"region": data_tools.ZONE_LABELS}).to_dict("records"),
        "caveat": "Exploratory country-clustered associations. Participation evidence is incomplete and strength uncertainty is not fully propagated.",
    }
    zone_results = publication.assemble_zones(
        point_frames, zone_inference, zone_adjustments, zone_sensitivities, strength_results,
        {"discrepancy": discrepancy_prediction, "bout": bout_prediction}, zone_federations,
        point_audit, event_dictionary, {**checks, "seed_reproducibility": reproducibility_passed})
    public_results = publication.build_snapshot(
        project_root, raw_bouts, bouts, variant_fits, variant_leagues, primary_summary,
        noise_draws, beta_comparison, rho_comparison, zone_results, confederation_summary, biography_coverage)
    report_text = publication.research_report(public_results)
    return public_results, report_text


@app.cell
def interpretation(
    beta_comparison,
    rho_comparison,
    strength_models,
    zone_inference,
):
    _c = zone_inference["contrasts"]
    _reward = next(row for row in strength_models if row["outcome"] == "zonal_points")
    _beta_p = beta_comparison.get("omnibus")
    _rho_p = rho_comparison.get("omnibus")
    _beta_label = f"{_beta_p['p']:.3f}" if _beta_p else "suppressed"
    _rho_label = f"{_rho_p['p']:.3f}" if _rho_p else "suppressed"
    _noise_statement = ("Neither overall test rejects equality at 5%; this does not establish equivalence."
                        if _beta_p and _rho_p and min(_beta_p["p"], _rho_p["p"]) >= .05
                        else "Read the overall tests with their intervals and validity diagnostics.")
    mo.md(f"""
    ### What the current results support

    - **Noise:** beta omnibus p = **{_beta_label}**; correlation omnibus p = **{_rho_label}**.
      {_noise_statement}
    - **Zonal-point mechanism:** the tie-controlled contrast changes from
      **{_c['D_total_ties']['estimate']:.3f}** to **{_c['D_without_zonal']['estimate']:.3f}**;
      mechanical reduction **{_c['fraction_reduction']['estimate']:.1%}**.
    - **Comparable non-zonal strength:** reward contrast **{_reward['estimate']:+.1f} points**,
      95% country-clustered interval **[{_reward['low']:.1f}, {_reward['high']:.1f}]**.
    - **Prediction:** interpret held-out discrepancy and bout performance separately.
      A ranking-system discrepancy is not athlete ability or a future ranking forecast.

    The geographic hypothesis remains exploratory. Entry selection, quotas, incomplete
    event coverage, missing biographies and name matching limit interpretation.
    Independent seasons and historical ranking snapshots are the next replication data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. Review and export

    The report below is a concise research draft with tables. The public snapshot is
    built from this run's 36 fits and primary four-zone results, without reading any
    previous snapshot. No raw biography records, licenses or birth dates are included.

    **Fantasy convention in the public app:** same-league probabilities use
    $\operatorname{logistic}(2eta(s_A-s_B))$. Across leagues, use
    $z=2eta(s-ar s)$ and $\operatorname{logistic}(z_A-z_B)$. Aligning disconnected
    league means is an entertainment convention, not an empirical common ability scale.

    Use the button to write the report, cleaned local tables and public snapshot.
    Publishing to Pages remains a separate Git commit/deployment step.
    """)
    return


@app.cell
def export_controls(report_text):
    export_results = mo.ui.run_button(label="Write reviewed analysis outputs", kind="success")
    mo.vstack([
        mo.download(report_text.encode("utf-8"), filename="fencing-research-report.md", label="Download report draft"),
        mo.accordion({"Read the report draft": mo.md(report_text)}),
        export_results,
    ])
    return (export_results,)


@app.cell
def write_outputs(
    event_dictionary,
    export_results,
    point_tables,
    project_root,
    public_results,
    publication,
    report_text,
):
    if export_results.value:
        output_paths = publication.write_exports(project_root, public_results, report_text, point_tables, event_dictionary)
        export_message = "Saved the report, public snapshot and cleaned local tables. No raw biographies were published."
    else:
        output_paths = {}
        export_message = "Analysis complete. Review the results above; use the export button when ready to write outputs."
    mo.md(export_message)
    return


if __name__ == "__main__":
    app.run()
