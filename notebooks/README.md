# Research notebook

`research_analysis.py` is one executable research narrative. It runs in reading
order from local inputs, with the four championship zones primary throughout.

1. Input audits, point reconciliation and cohort definition.
2. Six-category beta and FIE-correlation comparisons.
3. Four-zone discrepancy, paired zonal-points ablation and adjusted associations.
4. Rewards at comparable non-zonal SpringRank strength, with support checks.
5. Country-grouped discrepancy prediction and chronological bout prediction.
6. Age/participation, top-200, non-zonal-bout and confederation sensitivities.
7. Validation, seeded reproducibility and interpretation.
8. Review and a single export step.

## Run

Use Python 3.12 and the inputs described in `DATA.md`, including the enriched
biography CSV at `data/fie_biographies/athletes_enriched.csv`.

```sh
python -m pip install -r requirements-analysis.txt
python -m marimo edit notebooks/research_analysis.py
```

The notebook executes automatically. Settings are together near the top. The
primary protocol is senior events/minimum five bouts, 2,000 tournament draws,
seed 42; other cohorts belong to the sensitivity section. Bootstrap functions
resume only matching input-keyed checkpoints in `data/research/`. Prediction
models and ordinary fits run in notebook cells.

There are no independent old/new regional pipelines or intermediate report
writers. Five-confederation results are a secondary description from the network
bootstrap. The snapshot is built from the run's fits, without reading a previous
`analysis.json`.

## Review and export

The last section provides a report preview/download and one **Write reviewed
analysis outputs** button. It writes:

- `pages/public/analysis.json`: schema-3 actual results for the browser app.
- `Analysis/research_report.md` and a compatible `.txt`: one concise local draft.
- `data/research/clean_rankings/`: cleaned local rankings and event dictionary.

Raw inputs, biographies, licenses and birth dates are not published. The report
stays ignored for manual review. Publishing remains a separate Git commit/Pages
deployment. The original exploratory notebook is preserved.

## Published implementations

The notebook states research choices and owns every statistical run. Focused
helpers keep its cells short:

- `src/research.py`: network fitting, inference and prediction implementations.
- `src/ranking_points.py`: paired tournament resampling and point comparisons.
- `src/notebook_data.py`: parsing, classification and computed-result summaries.
- `src/zonal_strength.py`: common support and conditional reward models.
- `src/research_publication.py`: snapshot/report formatting and the export writer.

These modules perform no analysis on import and have no analysis CLI. GitHub Pages
serves the separate `pages/fencing_app.py` viewer of the committed results.
