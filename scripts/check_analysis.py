"""Check that the snapshot solver agrees with the original ranking implementation."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'scripts'))
from export_public_snapshot import fit_scores

spec = importlib.util.spec_from_file_location('original_springrank', root/'src/ranking_tools/springrank.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
matrix = np.array([[0,8,6,7], [2,0,6,5], [4,4,0,7], [3,5,3,0]], dtype=float)
names = ['A','B','C','D']
bouts = pd.DataFrame([{'Fencer A': names[i], 'Fencer B': names[j], 'Winner': names[i]}
                     for i in range(4) for j in range(4) for _ in range(int(matrix[i,j]))])
fitted, summary = fit_scores(bouts, 1)
expected = original.SpringRank().fit(matrix).ranks
actual = fitted.set_index('name').loc[names,'score'].to_numpy()
np.testing.assert_allclose(actual, expected, atol=1e-5)
assert summary['retained_bouts'] == int(matrix.sum())
assert summary['equation_residual'] < 1e-10
# A separate smaller component must not receive an arbitrary comparable offset.
disconnected = pd.concat([bouts, pd.DataFrame([{'Fencer A': 'E', 'Fencer B': 'F', 'Winner': 'E'}])], ignore_index=True)
kept, summary = fit_scores(disconnected, 1)
assert set(kept['name']) == set(names) and summary['components_before_filter'] == 2
print('Snapshot solver matches the original SpringRank objective and isolates disconnected components.')
