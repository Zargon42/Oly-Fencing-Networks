# Oly-Fencing-Networks

Analysis of Olympic fencing results using network-based ranking methods.

## Structure

```
Womens foil olympics/                       # Olympic women's foil bout, bio, ranking & tournament data
CoC dataset/                                 # athlete / DE / pool data from every event during the season 2025/26, all weapons, all categories
src/
  ranking_tools/                             # SpringRank & Bradley-Terry ranking implementations
  pysbm/                                     # vendored stochastic block model library
  tools.py, cv_tools.py, plot.py             # helper & plotting utilities
Ranking from pairwise comparisons_CoC_data.py  # ranking pipeline (Marimo notebook)
requirements.txt
```
## Next Steps
- pull official FIE rankings for each category and compare to calculated scores

## Acknowledgments

The ranking utilities in `src/ranking_tools/` are adapted from the code in
Caterina De Bacco's repository for the course *Probabilistic Inference in
Networks*:

<https://github.com/cdebacco/ds_prob_inf_net>

The adapted modules are distributed under the GNU General Public License
version 3. See `LICENSE` for the complete license text. `bradley_terry.py`
contains local modifications; the other ranking modules are retained as
adapted source unless their files state otherwise.

The primary analysis notebook also credits Professor De Bacco and the
fencing dataset contributor Andrew Fischl, known as CyrusOfChaos.

## License

This repository is distributed under the GNU GPL v3.0. Dependencies retain
their own licenses; see their respective project documentation.
