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
assert '<svg' in definitions['noise_panel'].text
assert '<svg' in definitions['differences_panel'].text
assert '<svg' in definitions['zonal_panel'].text
assert '5,303' in definitions['zonal_panel'].text
assert definitions['zonal_evidence']['regional_level'] == 'four_championship_zones'
assert 'Asia-Oceania' in definitions['zonal_evidence']['zone_labels']
chance = definitions['fantasy_chance']
assert chance(.3, 1.4, .3, 1.4) == .5
assert abs(chance(.3, 1.4, -.2, 1.8) + chance(-.2, 1.8, .3, 1.4) - 1) < 1e-12
assert 0 <= definitions['fantasy_probability'] <= 1
print('Public notebook ran successfully with actual senior women\'s foil results.')
