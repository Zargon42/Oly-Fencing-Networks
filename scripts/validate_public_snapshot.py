"""Validate the public data artifact without access to raw datasets or SciPy."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def validate(path):
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    assert snapshot['schema_version'] == 2
    assert snapshot['data_kind'] == 'actual_research_outputs'
    assert snapshot['model'] == 'SpringRank'
    assert snapshot['audit']['raw_bouts'] >= snapshot['audit']['fenced_valid_bouts'] > 0
    assert len(snapshot['variants']) == 36
    assert len({(v['population'], v['minimum_bouts'], v['category']) for v in snapshot['variants']}) == 36
    athletes = snapshot['athletes']
    research = snapshot['research']
    assert research['bootstrap']['replicates'] == 2000
    assert research['bootstrap']['seed'] == 42
    assert len(research['primary']) == 6
    for key in ['beta', 'rho']:
        result = research[key]
        assert result['requested'] == 2000
        assert len(result['intervals']) == 6 and len(result['pairs']) == 15
        assert result['formal_inference_enabled'] == (result['valid_fraction'] >= .9)
        for row in result['pairs']:
            assert row['low'] <= row['high']
            assert (row['p_holm'] is not None) == result['formal_inference_enabled']
    assert len(research['region_intervals']) == 30
    assert research['discrepancy_prediction']['metrics']
    assert research['bout_prediction']['status'] == 'complete'
    zonal = research['zonal_points']
    assert zonal['status'] == 'complete'
    assert len(zonal['reconciliation']) == 6
    assert all(row['athletes'] == row['reconciled'] and row['unreconciled'] == 0 and row['max_absolute_error'] == 0
               for row in zonal['reconciliation'])
    assert sum(row['athletes'] for row in zonal['reconciliation']) == 5303
    assert len(zonal['events']) == sum(row['events'] for row in zonal['reconciliation'])
    assert len({(r['category'], r['clean_header']) for r in zonal['events']}) == len(zonal['events'])
    assert zonal['bootstrap']['replicates'] == 2000 and zonal['bootstrap']['seed'] == 42
    assert zonal['bootstrap']['joint_valid'] <= 2000
    assert all(zonal['validation'].values())
    assert len(zonal['correlations']) == 6 and len(zonal['regional_intervals']) == 15
    for item in zonal['contrasts'].values():
        assert item['low'] <= item['high']
        assert item['formal_inference_enabled'] == (item['valid'] >= 1800)
        assert (item['p'] is not None) == item['formal_inference_enabled']
    assert abs(zonal['contrasts']['D_without_zonal']['estimate'] - zonal['contrasts']['D_total_ties']['estimate']
               - zonal['contrasts']['D_change']['estimate']) < 1e-10
    assert len({(a['category'], a['name']) for a in athletes}) == len(athletes)
    assert all(a['name'] and a['category'] in snapshot['category_labels'] for a in athletes)
    for variant in snapshot['variants']:
        rows = variant['rows']
        assert math.isfinite(variant['beta']) and variant['beta'] >= 0
        assert len(rows) == variant['fencers'] > 0
        assert len({r[0] for r in rows}) == len(rows)
        assert all(0 <= r[0] < len(athletes) and athletes[r[0]]['category'] == variant['category'] for r in rows)
        assert all(len(r) == 5 and math.isfinite(r[1]) and r[2] >= 1 and r[3] > 0 for r in rows)
        assert all(r[4] is None or isinstance(r[4], int) and r[4] > 0 for r in rows)
        assert sum(r[3] for r in rows) == 2 * variant['retained_bouts']
        assert abs(sum(r[1] for r in rows) / len(rows)) < 1e-8
        assert variant['equation_residual'] < 1e-7
        assert variant['input_bouts'] >= variant['retained_bouts']
        for label, matches in [('all', [r for r in rows if r[4] is not None]),
                               ('top200', [r for r in rows if r[4] is not None and r[4] <= 200])]:
            result = variant['comparison'][label]
            assert result['n'] == len(matches)
            if len(matches) >= 3:
                frame = pd.DataFrame(matches, columns=snapshot['row_columns'])
                recomputed = float(frame['score'].rank().corr(frame['fie_rank'].rank()))
                assert abs(recomputed - result['spearman']) < 1e-6, (variant['category'], label, recomputed, result)
    for source in snapshot['source_files']:
        assert len(source['sha256']) == 64
    serialized = json.dumps(snapshot).lower()
    assert not any(key in serialized for key in ['"datebirth"', '"dob"', '"license"', '"birth_date"'])
    print(f'Validated all 36 real analysis variants ({path.stat().st_size:,} bytes).')
    print('Snapshot SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', nargs='?', type=Path, default=Path(__file__).resolve().parents[1]/'pages/public/analysis.json')
    validate(parser.parse_args().snapshot)
