import json
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_GLOB = ROOT / 'artifacts' / 'simulation_artifact_*.json'

thresholds = [0.8, 0.5, 0.1]

files = sorted(glob.glob(str(ARTIFACT_GLOB)))
if not files:
    print('No artifact files found in', ARTIFACT_GLOB)
    raise SystemExit(1)

for path in files:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    meta = data.get('metadata', {})
    cfg = data.get('configuration', {})
    market_cfg = cfg.get('market', {})
    index_name = meta.get('index_name', Path(path).stem)
    initial = market_cfg.get('initial_value', None)
    num_timesteps = meta.get('num_timesteps', None)
    percentiles = data.get('market_percentiles', {})
    p05 = percentiles.get('p05', [])
    p50 = percentiles.get('p50', [])
    p95 = percentiles.get('p95', [])

    def final_val(arr):
        return arr[-1] if arr else None

    final_p05 = final_val(p05)
    final_p50 = final_val(p50)
    final_p95 = final_val(p95)

    def days_to_threshold(arr, thresh_value):
        for i, v in enumerate(arr):
            try:
                if v <= thresh_value:
                    return i  # 0-based timestep index
            except Exception:
                continue
        return None

    print('Index:', index_name)
    print(' - initial value:', initial)
    print(' - timesteps:', num_timesteps)
    print(' - final p05: {:.6g}'.format(final_p05) if final_p05 is not None else ' - final p05: N/A')
    print(' - final p50: {:.6g}'.format(final_p50) if final_p50 is not None else ' - final p50: N/A')
    print(' - final p95: {:.6g}'.format(final_p95) if final_p95 is not None else ' - final p95: N/A')

    for label, arr in [('p05', p05), ('p50', p50), ('p95', p95)]:
        print(f' - days to thresholds for {label}:')
        for t in thresholds:
            if initial is None or not arr:
                print(f'    {int((1-t)*100)}%: N/A')
                continue
            thresh_val = initial * t
            d = days_to_threshold(arr, thresh_val)
            if d is None:
                print(f'    {int((1-t)*100)}%: Not reached')
            else:
                print(f'    {int((1-t)*100)}%: day {d} (timestep index)')
    # plot path guess
    safe_name = index_name.replace('/', '_').replace(' ', '_')
    plot_path = ROOT / 'graphs' / f'simulation_results_{safe_name}.png'
    print(' - plot:', plot_path)
    print(' - artifact file:', path)
    print('')

print('Parsed', len(files), 'artifact(s).')
