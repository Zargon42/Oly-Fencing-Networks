"""Run the public notebook locally and check its default real-data view."""
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_fencing_app', root/'pages/fencing_app.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
outputs, definitions = module.app.run()
assert definitions['population'].value == 'Senior events'
assert definitions['category'].value == 'WF'
assert definitions['minimum_bouts'].value == 5
assert len(definitions['rankings']) == definitions['selected_variant']['fencers']
assert len(definitions['comparison']) == definitions['selected_variant']['comparison']['all']['n']
assert abs(definitions['rank_correlation'] - definitions['selected_variant']['comparison']['all']['spearman']) < 1e-6
assert definitions['snapshot']['data_kind'] == 'actual_research_outputs'
assert sum('<svg' in str(getattr(output, 'text', '')) for output in outputs) == 2, 'Both charts must render actual SVG content'
print('Public notebook ran successfully with actual senior women\'s foil results.')
