# Data sources and reproducibility

## Run the public app without raw data

The repository includes a compact derived-results artifact at
`pages/public/analysis.json`. It contains real outputs from the recorded 2025/26
bouts and archived ranking comparisons. This file is sufficient to run the public
app or build the GitHub Pages site. No data upload is required from visitors.

The artifact is intentionally public: it includes athlete names, recorded country
labels, scores, retained bout counts, and matched archived official ranks. It omits
raw bout rows and individual biography records. This is a selective publication of
analysis outputs, not a live FIE ranking service or a copy of the full source datasets.

## Rebuild the snapshot locally

Expected paths, relative to the repository root:

| Input | Purpose |
| --- | --- |
| `CoC dataset/Pools.csv` | Pool bout outcomes |
| `CoC dataset/DEs.csv` | Direct-elimination bout outcomes |
| `rankings/ME-detailed-ranking-2026.csv` | Archived men's epee standings |
| `rankings/WE-detailed-ranking-2026.csv` | Archived women's epee standings |
| `rankings/MF-detailed-ranking-2026.csv` | Archived men's foil standings |
| `rankings/WF-detailed-ranking-2026.csv` | Archived women's foil standings |
| `rankings/MS-detailed-ranking-2026.csv` | Archived men's sabre standings |
| `rankings/WS-detailed-ranking-2026.csv` | Archived women's sabre standings |

The source repository credits Andrew Fischl, known as CyrusOfChaos, for the bout
dataset. Use the original supplied 2025/26 tables; no stable independent download
URL for those exact inputs is documented here. Where inputs are already present in
your checkout, no upload or new download is needed. Access through the original
dataset provider may be necessary to reproduce from raw data elsewhere.

FIE profiles and rankings originate at [fie.org](https://fie.org/athletes). Ranking
CSVs used for this snapshot were available in the project's archived repository
snapshot `63f3f854fc84ca55d4e1fb946b0895802f60e619`, under `rankings/`. Their original
retrieval timestamp is not recorded. Fetching current standings later will not
reproduce that historical reference. SHA256 hashes in the public artifact identify
the exact local files used. If ranking CSVs are absent, snapshot generation can
still fit the models, but no official-rank comparisons will be available.

The exporter requires bout columns `Fencer A`, `Fencer B`, `Winner`, `Country A`,
`Country B`, `Weapon`, `Gender`, `Age Category`, `Bout Status`, and `Event Date`.
Ranking tables require `Rank` and `Name`. Names are matched after case, accent,
punctuation, and token-order normalization. Ambiguous normalized names are not
matched. This does not replace license-based identity verification.

## Model and filtering

The exporter solves the same SpringRank objective as the local research notebook,
using a sparse direct solver. `scripts/check_analysis.py` compares it with the
repository's original implementation. Six weapon/gender categories are fitted
separately; the public snapshot avoids treating disconnected men's and women's
networks as comparable scores. It excludes administrative or unavailable outcomes.
Minimum-participation settings are applied before filtering opponents, then the
largest weakly connected component is retained. Scores are mean-centered per fit.

Invalid dates are counted and reported without guessing replacements. Since the
snapshot is non-temporal, valid fenced outcomes with invalid dates remain eligible.
Do not use this dataset for chronological forecasting until those dates are resolved.

## Biography enrichment

The local research notebook imports published FIE biographies into
`data/fie_biographies/`. Each profile JSON records its source URL and retrieval time.
The enriched CSV preserves original athlete fields and records unresolved matches.
These caches are ignored and are not used by or bundled into the public app.

## Attribution

- Bout data: Andrew Fischl / CyrusOfChaos, as credited in the original project.
- Official standings and profiles: International Fencing Federation (FIE).
- Adapted ranking utilities: Caterina De Bacco's
  [Probabilistic Inference in Networks](https://github.com/cdebacco/ds_prob_inf_net) material.
- Project code: GPL v3, as stated in `LICENSE`. That code license does not establish
  a license for separately sourced data.
