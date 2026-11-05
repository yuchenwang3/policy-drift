#!/usr/bin/env python3
"""Summarize saved reduction/Adam-state results without running the model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lab import dump_json, write_csv


def distribution(logits):
    z = logits.astype(np.float64)
    q = np.exp(z-z.max(axis=-1, keepdims=True))
    return q/q.sum(axis=-1, keepdims=True)


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    args = parser.parse_args()
    root, figures = args.results, args.figures
    figures.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root/'summary.json').read_text())
    rows = []
    for r in summary['moment_comparisons']:
        native, zero, pair = r['native_gradient_branch'], r['zero_gradient_control'], r['native_vs_zero_policy']
        rows.append({'label': r['label'], 'warmup_steps': r['warmup_steps'],
                     'state_kind': r['state_kind'], 'adam_step_before': r['adam_step_before'][0],
                     'native_gradient_l2': native['gradient_l2'],
                     'zero_gradient_update_l2': zero['update_l2'],
                     'native_gradient_update_l2': native['update_l2'],
                     'native_minus_zero_update_l2': r['native_minus_zero_update_l2'],
                     'native_minus_zero_changed_parameter_coordinates': r['native_minus_zero_changed_parameter_coordinates'],
                     'native_vs_zero_policy_kl': pair['kl_mean'],
                     'native_vs_zero_argmax_changes': pair['argmax_different_prefixes'],
                     'zero_vs_before_policy_kl': zero['kl_mean'],
                     'native_vs_before_policy_kl': native['kl_mean'],
                     'zero_before_argmax_ids': json.dumps(zero['before_argmax_ids']),
                     'zero_after_argmax_ids': json.dumps(zero['after_argmax_ids']),
                     'native_after_argmax_ids': json.dumps(native['after_argmax_ids'])})
    write_csv(root/'paired-summary.csv', rows)
    states = []
    for p in sorted(root.glob('adam-state-step-*.json'), key=lambda p: int(p.stem.split('-')[-1])):
        states.append({'warmup_steps': int(p.stem.split('-')[-1]), **json.loads(p.read_text())})
    output = {'actual_optimizer_steps': summary['actual_optimizer_steps'],
              'warmup_actual_optimizer_steps': summary['warmup_actual_optimizer_steps'],
              'reduction_trial_count': summary['reduction_trial_count'],
              'original_argmax_pair': summary['original_argmax_pair'],
              'same_transition_minimum': summary['minimum_same_original_transition'],
              'any_transition_minimum': summary['minimum_any_transition'],
              'paired_comparisons': rows, 'accumulated_state': states,
              'stable_controls': summary['minimum_stable_control'],
              'scope': {'minimum': summary['minimality_scope'],
                        'state_origin': 'actual hard-label Adam updates on the original eight fixed examples',
                        'moment_transplant': 'same original weights, moments from warmup; off-trajectory diagnostic',
                        'effect_measure': 'native residual branch minus explicit zero-gradient branch from cloned common state',
                        'generalization': 'one model, one fixture set, checkpoints 16 and 128; no RL'}}
    dump_json(root/'findings.json', output)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    case_labels = list(summary['probe_cases'])
    fig, axes = plt.subplots(1, len(case_labels), figsize=(6.2*len(case_labels), 4.5),
                             constrained_layout=True, squeeze=False)
    for i, label in enumerate(case_labels):
        r = next(r for r in summary['moment_comparisons'] if r['label'] == label+'-initial-fresh')
        data = np.load(root/f'paired-{label}-initial-fresh.npz')
        old, native = distribution(data['before'])[0], distribution(data['native_after'])[0]
        old_id, new_id = int(old.argmax()), int(native.argmax())
        names = [r['native_gradient_branch']['before_argmax_text'][0], r['native_gradient_branch']['after_argmax_text'][0]]
        x = np.arange(2)
        axes[0, i].bar(x-.18, old[[old_id, new_id]], .36, label='Before / exact-zero control', color='#417ba0')
        axes[0, i].bar(x+.18, native[[old_id, new_id]], .36, label='After native FP32 soft CE + Adam', color='#c76b31')
        axes[0, i].set_xticks(x, [repr(t) for t in names])
        case = summary['probe_cases'][label]
        axes[0, i].set(title=f'{label} | {len(case["rows"])} prefix, {sum(len(x["input_ids"]) for x in case["rows"])} tokens',
                       ylabel='Probability of the two competing tokens')
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(axis='y', alpha=.15)
    fig.suptitle('Reduced same-model self-distillation counterexamples | fresh Adam, lr=1e-5, eps=1e-8, T=4')
    save(fig, figures/'minimum-counterexamples')
    checkpoints = sorted(set(r['warmup_steps'] for r in rows if r['warmup_steps']))
    fig, axes = plt.subplots(len(case_labels), 2, figsize=(12, 4.3*len(case_labels)),
                             constrained_layout=True, squeeze=False)
    for index, label in enumerate(case_labels):
        fresh = next(r for r in rows if r['label'] == label+'-initial-fresh')
        points = [fresh] + [next(r for r in rows if r['label'] == f'{label}-original-warm-{step}') for step in checkpoints]
        xs = [0]+checkpoints
        for axis, metric, title in [(axes[index, 0], 'native_minus_zero_update_l2', 'Extra stored parameter update'),
                                    (axes[index, 1], 'native_vs_zero_policy_kl', 'Extra policy change: KL(zero || native)')]:
            axis.plot(xs, [r[metric] for r in points], '-o', color='#77549a', label='Original weights, accumulated moments')
            axis.axhline(fresh[metric], ls=':', color='#596a79', label='Fresh Adam at original weights')
            actual = [fresh] + [next(r for r in rows if r['label'] == f'{label}-current-warm-{step}') for step in checkpoints]
            cold = [fresh] + [next(r for r in rows if r['label'] == f'{label}-current-fresh-{step}') for step in checkpoints]
            axis.plot(xs, [r[metric] for r in actual], '-s', color='#417ba0', label='Actual warmup weights + moments')
            axis.plot(xs, [r[metric] for r in cold], '--^', color='#c76b31', label='Same warmup weights, reset moments')
            axis.set_yscale('symlog', linthresh=1e-16 if 'kl' in metric else 1e-10)
            axis.set_xticks(xs)
            axis.set(xlabel='Number of actual supervised warmup steps', ylabel=metric,
                     title=f'{label}: {title}')
            axis.grid(alpha=.15)
            axis.legend(fontsize=7, loc='best')
    fig.suptitle('Residual effect uses a paired zero-gradient control | common parameters and cloned Adam state')
    save(fig, figures/'adam-history-paired-effects')
    print(json.dumps({'status':'complete','comparisons':len(rows),'figures':2,
                      'actual_optimizer_steps':summary['actual_optimizer_steps']}))


if __name__ == '__main__':
    main()
