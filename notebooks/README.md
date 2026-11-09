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
