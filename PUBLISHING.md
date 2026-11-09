# Publish the real analysis on GitHub Pages

The public app is `pages/fencing_app.py`. It runs Python in each visitor's browser
through marimo and Pyodide. GitHub Pages serves its static export; no paid Python
server, API key, or backend storage is required. Internet access is needed to load
the browser runtime and Python packages. The first visit can take time to start.

## Public data

`pages/public/analysis.json` contains actual, reproducible SpringRank outputs.
There is no synthetic dataset. It includes athlete names and recorded country
labels, calculated scores, retained bout counts, archived FIE ranks when matched,
category summaries, input hashes, and the settings used. The public site necessarily
makes this compact snapshot downloadable. Raw bouts, biography caches, licenses,
birth dates, and the unreviewed report are not bundled in the Pages build.

There are 36 independently fitted selections: six weapon/gender categories,
senior-only or all recorded ages, and minimum participation of 1, 5, or 10 bouts.
Browser controls select and filter these saved results. They do not fit arbitrary
new models, fetch FIE profiles, or run the local background importer.

## Preview locally

From the repository root, using Python 3.12:

```sh
python -m pip install -r requirements-pages.txt
python scripts/validate_public_snapshot.py
python scripts/check_pages.py
python -m marimo check pages/fencing_app.py --select MW
python scripts/build_pages.py
python -m http.server 8000 --directory _site
```

Open `http://localhost:8000/`. This tests the actual browser app rather than a
server-backed notebook. Do not open the exported HTML using a `file://` URL.

To edit the public viewer locally:

```sh
python -m marimo edit pages/fencing_app.py
```

## Enable the free GitHub Pages deployment

1. Review the app and snapshot locally.
2. Commit and push `pages/`, `scripts/`, `requirements-pages.txt`,
   `requirements-analysis.txt`, `.github/workflows/pages.yml`, and the documentation.
3. In the GitHub repository, open **Settings > Pages**.
4. Set **Build and deployment > Source** to **GitHub Actions**.
5. Open **Actions > Publish fencing analysis to Pages > Run workflow** on `main`.
6. Wait for both build and deploy jobs to succeed. Use the URL reported by deploy.

For the current repository, the expected project-site URL is
`https://zargon42.github.io/Oly-Fencing-Networks/`; it is not live until deployment
succeeds. No custom domain is required. Subsequent relevant pushes to `main`
rebuild and deploy automatically. Pull requests build and validate without deploying.

The workflow exports only the public notebook and its `public/` assets. It does
not publish the repository root, `Analysis/`, or the local research notebook.

## Refresh the real results

Obtain the source inputs described in `DATA.md`, then run locally:

```sh
python -m pip install -r requirements-analysis.txt
python -m marimo edit notebooks/research_analysis.py
python scripts/validate_public_snapshot.py
python scripts/check_pages.py
```

Run the visible bootstrap/model controls and final export cells in the research
notebook. The notebook calls published helpers in `src/research.py`; all substantive
analysis runs are owned by notebook cells. The legacy schema-1 CLI exporter is not
the research workflow. See [notebook instructions](notebooks/README.md).

The schema-2 snapshot adds beta estimates, uncertainty, Holm comparisons, regional
summaries and prediction-validation outputs. The local report remains ignored.

Review the changed snapshot and commit `pages/public/analysis.json` to publish
the updated results. The Pages workflow builds from that committed artifact and
never scrapes websites or needs your ignored input folders.

## Git cleanup

Virtual environments, bytecode, marimo sessions, build output, raw downloads, and
local drafts are ignored. Existing tracked runtime artifacts and the ignored
archived `rankings/` inputs are removed from the Git index while leaving local
copies intact. The original `CoC dataset/` and `Womens foil olympics/` datasets
remain tracked. Removing files from the current index does not erase earlier Git history.

The public app does not need saved marimo session outputs: it loads its actual
snapshot and renders in the browser. If you later use molab static previews,
selectively allow only a reviewed session JSON for that purpose.

## References

- [marimo on GitHub Pages](https://docs.marimo.io/guides/publishing/github/)
- [marimo WebAssembly exports](https://docs.marimo.io/guides/exporting/webassembly_html/)
- [GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
