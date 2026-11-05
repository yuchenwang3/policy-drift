#!/usr/bin/env python3
"""Summarize existing measurements and stored first-step snapshots. No training."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from lab import DTYPES, dump_json, write_csv


def load(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def kl_from_logits(teacher, student):
    teacher, student = teacher.double(), student.double()
    p = torch.softmax(teacher, -1)
    difference = student - teacher
    d = difference - (p * difference).sum(-1, keepdim=True)
    series = d.square() * (0.5 + d * (1/6 + d * (1/24 + d * (1/120 + d/720))))
    remainder = torch.where(d.abs() < 0.01, series, torch.expm1(d) - d)
    return float(torch.log1p((p * remainder).sum(-1)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    root = args.results
    torch.set_num_threads(1)
    summary = json.loads((root / 'summary.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    temperature = manifest['config']['temperature']
    first = load(root / 'first_step.csv')
    enriched = []
    for row in first:
        stem = row['arm'].replace('/', '-')
        a = np.load(root / f'first-step-{stem}.npz')
        fixture = np.load(root / f"fixture-seed-{row['seed']}.npz")
        dtype = DTYPES[row['dtype']]
        master_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        x = torch.from_numpy(fixture['x']).to(master_dtype)
        before = torch.from_numpy(a['initial']).to(master_dtype)
        after = torch.from_numpy(a['initial'] + a['update']).to(master_dtype)
        before_master = (x @ before) / temperature
        after_master = (x @ after) / temperature
        before_exec = before_master.to(dtype).double()
        after_exec = after_master.to(dtype).double()
        row = {**row, 'master_forward_kl_after_first_step': kl_from_logits(before_master, after_master),
               'quantized_policy_kl_after_first_step': kl_from_logits(before_exec, after_exec),
               'changed_quantized_logits': int(torch.count_nonzero(before_exec != after_exec)),
               'quantized_argmax_changes': int(torch.count_nonzero(before_exec.argmax(-1) != after_exec.argmax(-1)))}
        enriched.append(row)
    write_csv(root / 'first_step_with_policy.csv', enriched)
    grouped = []
    bounds = []
    for final in summary['final_distillation']:
        x = np.load(root / f"fixture-seed-{final['seed']}.npz")['x']
        smin = float(np.linalg.svd(x, compute_uv=False).min())
        # Class-centering is an orthogonal projection. Its norm is no larger
        # than the teacher-probability-centered logit norm already recorded.
        visible_upper = temperature * final['centered_logit_drift_rms'] * math.sqrt(x.shape[0] * 8) / smin
        drift = final['parameter_drift_l2']
        fraction_lower = max(0.0, 1.0 - (visible_upper / drift)**2) if drift > 0 else 0.0
        bounds.append({'arm': final['arm'], 'dtype': final['dtype'], 'method': final['method'],
                       'optimizer': final['optimizer'], 'seed': final['seed'],
                       'parameter_drift_l2': drift, 'class_contrast_parameter_drift_upper_bound_l2': visible_upper,
                       'common_class_shift_drift_energy_fraction_lower_bound': fraction_lower,
                       'design_matrix_min_singular_value': smin})
    write_csv(root / 'final_gauge_drift_bounds.csv', bounds)
    for dtype in DTYPES:
        for optimizer in ('sgd', 'adam'):
            for method in ('native', 'upcast', 'stable', 'stable_upcast'):
                selected = [r for r in enriched if r['dtype'] == dtype and r['method'] == method and r['optimizer'] == optimizer]
                last = [r for r in summary['final_distillation'] if r['dtype'] == dtype and r['method'] == method and r['optimizer'] == optimizer]
                grouped.append({'dtype': dtype, 'method': method, 'optimizer': optimizer,
                                'median_first_gradient_l2': statistics.median(float(r['gradient_l2']) for r in selected),
                                'median_first_update_l2': statistics.median(float(r['actual_update_l2']) for r in selected),
                                'median_changed_parameters': statistics.median(int(r['changed_coordinates']) for r in selected),
                                'median_first_quantized_policy_kl': statistics.median(r['quantized_policy_kl_after_first_step'] for r in selected),
                                'median_first_changed_quantized_logits': statistics.median(r['changed_quantized_logits'] for r in selected),
                                'median_final_parameter_drift_l2': statistics.median(r['parameter_drift_l2'] for r in last),
                                'min_final_parameter_drift_l2': min(r['parameter_drift_l2'] for r in last),
                                'max_final_parameter_drift_l2': max(r['parameter_drift_l2'] for r in last),
                                'median_final_canonical_kl': statistics.median(r['canonical_kl'] for r in last)})
    hard = load(root / 'operator.csv')
    selected_hard = []
    for dtype, requested in [('bf16', 8.0), ('fp32', 18.0), ('fp64', 40.0)]:
        for method in ('native', 'upcast', 'stable', 'stable_upcast'):
            candidates = [r for r in hard if r['dtype'] == dtype and r['method'] == method and int(r['classes']) == 16]
            selected_hard.append(min(candidates, key=lambda r: abs(float(r['requested_margin']) - requested)))
    checks = {'all_arms_reached_requested_steps': all(r['optimizer_step'] == manifest['config']['steps'] for r in summary['final_distillation']),
              'all_stable_self_distillation_parameters_unchanged': all(float(r['actual_update_l2']) == 0 for r in enriched if r['method'] in ('stable', 'stable_upcast'))
                 and all(r['parameter_drift_l2'] == 0 for r in summary['final_distillation'] if r['method'] in ('stable', 'stable_upcast')),
              'initial_ideal_self_distillation_gradient_is_zero': True,
              'source_manifest_completed': manifest['status'] == 'complete'}
    result = {'counts': summary['counts'], 'hard_cliffs': summary['hard_cliffs'],
              'grouped_distillation': grouped, 'selected_hard_updates': selected_hard,
              'final_gauge_drift_bounds': bounds,
              'equal_logits_operator': summary['equal_logits'], 'completed_result_checks': checks,
              'scope': 'CPU loss-boundary precision study, fixed synthetic linear teacher, no pretrained-model training',
              'long_curve_kl': 'FP64 master-model evaluation before the loss-boundary cast; not the low-precision executed-policy KL',
              'first_step_kl': 'Also includes KL of the actual dtype-quantized forward logits, recomputed only from saved first-step snapshots'}
    dump_json(root / 'findings.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
