# Executable research

`research_analysis.py` is the source of the extended analysis: corrected beta
estimation, six-category tournament bootstrap comparisons, ranking discrepancies,
regional associations, held-out-country prediction and chronological bout prediction.
All substantive runs and exports are initiated by visible marimo cells. Published
helper implementations are in `src/research.py`; they contain no independent runner.

From the repository root, create a Python 3.12 environment and install
`requirements-analysis.txt`, then open:

```sh
python -m marimo edit notebooks/research_analysis.py
```

Supply the local bout files and archived rankings described in `DATA.md`, plus
`data/fie_biographies/athletes_enriched.csv` for the biography models. The notebook
shows the FIE region source, input audit and biography coverage. Run the two analysis
buttons; the bootstrap defaults to 2,000 replicates and seed 42. Its input-keyed
checkpoint is local in `data/research/`.

The final cells export the compact actual-results snapshot to
`pages/public/analysis.json` and an unreviewed local draft to
`Analysis/research_report.txt`. Raw biographies, licenses and dates of birth are not
published. GitHub Pages serves `pages/fencing_app.py`, a browser-only interactive
viewer of those results. It does not repeat the expensive native analysis per visitor.

The original exploratory notebook is preserved. The new notebook separates the
formal analysis from that exploratory history. The archived public snapshot remains
usable without downloading the original source data.

## Reconciled tournament points

The ranking-point cells repair reversed tournament headers and preserve an
original-to-clean event dictionary. They reproduce every published total using
FIE Organisation Rules, December 2025, o.108.1(a): best five World Cup/Grand Prix/
Satellite results plus the world/Olympic and zonal slots. Clean tables stay local
under `data/research/clean_rankings/`; original inputs are preserved.

The zonal-point counterfactual removes the separate zonal slot without promoting
a sixth ordinary result. It holds observed participation fixed and uses average
point ties before and after within the same matched cohort. It does not claim to
reproduce every official placement tie-break. Another 2,000 paired tournament
resamples (seed 42) quantify regional-contrast and correlation changes. The final
point export extends the public snapshot and local report. Run the point cells
after the core research cells when reproducing the full export.

## Four-zone primary hypothesis

The final four-zone cells combine Asia and Oceania for the shared championship
points pool. Confederation membership is retained separately; five-confederation
results are secondary. Run these cells and their final export after the earlier
ranking-points section. Another 2,000 paired tournament draws (seed 42) give the
correct four-zone contrasts and intervals.

The complementary strength-conditioned test fits SpringRank without zonal bouts,
then compares recorded zonal rewards among evidenced participants within common
four-zone strength support. Country-clustered regressions use category-specific
cubic strength-percentile curves, category, age and hand. These are exploratory
conditional associations; missing participation evidence and estimated strength
limit causal interpretation. The public app shows coverage and support exclusions.
