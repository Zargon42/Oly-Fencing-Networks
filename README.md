# Oly-Fencing-Networks

Network-based analysis of international fencing results using SpringRank, with an
interactive marimo app designed for free hosting on GitHub Pages.

The active analysis uses recorded 2025/26 pool and direct-elimination bouts across
all six weapon/gender categories. Historical women's foil data is also retained
as a separate source. The public app displays **actual analysis outputs**, not
synthetic examples, and compares the inferred ordering with archived FIE ranks.

## Public app

The browser app lets visitors select the category, senior-only or combined-age
events, and minimum participation of 1, 5, or 10 bouts. They can inspect rankings,
search athletes, compare FIE ordering, view charts, and download the displayed table.
It now leads with six-category noise comparisons, tournament-bootstrap uncertainty,
regional ranking discrepancies and held-out prediction results. Fantasy matchups
include an explicit hypothetical comparison caveat for disconnected categories.
The FIE-differences section also shows the reconciled zonal-points counterfactual:
all 5,303 supplied totals match the counting rule, with paired uncertainty for
the change in regional discrepancies and SpringRank agreement.
The primary comparison uses the four championship zones, combining Asia and
Oceania. A complementary model compares zonal rewards at comparable SpringRank
strength estimated without zonal bouts. Five-confederation results are secondary.

`pages/public/analysis.json` contains the compact derived-results snapshot needed
by the app. Visitors do not need to upload source data. Raw bouts, FIE biography
caches, and the unreviewed report are not included in the Pages build.

See [PUBLISHING.md](PUBLISHING.md) for the one-time GitHub Pages setup and local
preview instructions. The expected URL after successful deployment is
<https://zargon42.github.io/Oly-Fencing-Networks/>.

## Quick start

Use Python 3.12. From the repository root:

```sh
python -m pip install -r requirements-pages.txt
python -m marimo edit pages/fencing_app.py
```

To check and export the actual browser version:

```sh
python scripts/validate_public_snapshot.py
python scripts/check_pages.py
python scripts/build_pages.py
python -m http.server 8000 --directory _site
```

Open `http://localhost:8000/`. The app runs Python in the browser and needs internet
access initially to load Pyodide and dependencies. No paid hosting or backend is required.

## Reproduce the analysis

Data provenance, input schemas, exact input hashes, and the acquisition limitations
are described in [DATA.md](DATA.md). With the local source files available:

```sh
python -m pip install -r requirements-analysis.txt
python -m marimo edit notebooks/research_analysis.py
python scripts/validate_public_snapshot.py
```

The formal research notebook is `notebooks/research_analysis.py`; see
[its instructions](notebooks/README.md). It runs one complete analysis in reading
order; review the results and use the single final export button. The schema-3
snapshot is constructed from those fits, without reading an earlier snapshot.
The legacy CLI exporter must not overwrite it.

The original exploratory notebook remains `Ranking from pairwise comparisons_CoC_data.py`.
It supports local exploration and resumable FIE biography enrichment. Its original
broader dependency list is retained in `requirements.txt`; install those additional
dependencies if you use the full historical research environment. The lightweight
Pages app has its own inline dependency metadata and `requirements-pages.txt`.

## Interpretation

Higher SpringRank scores indicate stronger observed performance. Official ranks
use the opposite direction, so agreement produces a negative score-versus-rank
correlation. The public app reports that correlation within a single category and
population. Scores from independent fits do not share an absolute ability scale.

Agreement with standings that may use the same events is concurrent validation,
not evidence of future-match accuracy. Participation filters change the evaluation
population. Name-based matching, incomplete data coverage, and invalid event dates
are recorded limitations. The app exposes methods and scope alongside the results.

## Structure

```text
pages/fencing_app.py                 Browser-compatible marimo app
pages/public/analysis.json           Compact real analysis snapshot
scripts/export_public_snapshot.py   Rebuild results from local source inputs
scripts/validate_public_snapshot.py Check the published snapshot
scripts/check_pages.py              Run the public app locally as a smoke check
scripts/check_analysis.py           Compare snapshot solver with original SpringRank
.github/workflows/pages.yml         Validate, export, and deploy to GitHub Pages
src/ranking_tools/                  Adapted SpringRank and Bradley-Terry utilities
CoC dataset/                        Original season inputs, where available locally
Womens foil olympics/               Separate historical source data
DATA.md                             Data provenance and reproduction instructions
PUBLISHING.md                       Free hosting and refresh workflow
```

The `.gitignore` excludes local environments, caches, downloaded ranking and
biography data, scratch work, build artifacts, and unreviewed report directories.

## Acknowledgments

The fencing dataset is credited to Andrew Fischl, known as CyrusOfChaos.

The utilities in `src/ranking_tools/` are adapted from Caterina De Bacco's
[Probabilistic Inference in Networks course repository](https://github.com/cdebacco/ds_prob_inf_net).
`bradley_terry.py` contains local modifications. The snapshot exporter solves the
same SpringRank objective with a sparse direct solver, checked against the adapted
implementation. Official ranking and biography information originates from
the [International Fencing Federation](https://fie.org/athletes).

## License

Project code is distributed under GNU GPL v3. See [LICENSE](LICENSE). Dependencies
and separately sourced datasets retain their own terms; the code license does
not establish a license for those data sources.
