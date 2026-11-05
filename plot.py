#!/usr/bin/env python3
"""Render the saved measurements; never reruns an optimizer or model."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS = {'native': '#ce622b', 'upcast': '#3079aa', 'stable': '#238265', 'stable_upcast': '#8051a7'}
LABELS = {'native': 'Native CE', 'upcast': 'Native CE in FP32*', 'stable': 'Stable in input dtype', 'stable_upcast': 'Stable in FP32*'}
DTYPES = ['bf16', 'fp16', 'fp32', 'fp64']


def load(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def values(rows, key):
    return np.array([float(r[key]) for r in rows])


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=190, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def legends(fig, optimizers=False):
    handles = [Line2D([0], [0], color=COLORS[m], label=LABELS[m], lw=2,
                      ls='--' if m == 'upcast' else '-') for m in COLORS]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.035), borderaxespad=0)


def operator_plot(hard, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    for ax, dtype in zip(axes.flat, DTYPES):
        for method in COLORS:
            rows = [r for r in hard if r['dtype'] == dtype and r['method'] == method and int(r['classes']) == 16]
            ax.plot(values(rows, 'actual_margin'), np.maximum(values(rows, 'grad_target_relative_error'), 1e-18),
                    color=COLORS[method], lw=1.9, ls='--' if method == 'upcast' else '-', label=LABELS[method])
        ax.set(title=dtype, yscale='log', ylim=(1e-18, 3), xlim=(0, 60))
        ax.axhline(1, color='#555555', lw=0.8, ls=':')
        ax.grid(alpha=0.15)
    for ax in axes[:, 0]:
        ax.set_ylabel('Target-gradient relative error')
    for ax in axes[1]:
        ax.set_xlabel('Actual aggregate logit margin (nats)')
    fig.suptitle('Operator error on identical quantized inputs | 16 classes | 80-digit reference', fontsize=13)
    legends(fig)
    fig.text(0.5, -0.105, 'Zero error is displayed at 1e-18. Relative error 1 means the target gradient is lost.\n* FP64 inputs remain FP64; gradients still cross the original input dtype boundary.', ha='center', fontsize=9)
    save(fig, out / 'operator-gradient-error')

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    for ax, dtype in zip(axes.flat, DTYPES):
        rows = [r for r in hard if r['dtype'] == dtype and r['method'] == 'native' and int(r['classes']) == 16]
        ax.plot(values(rows, 'actual_margin'), np.abs(values(rows, 'ref_grad_target')), color='#333333', lw=2.6, ls=':', label='Reference')
        for method in COLORS:
            rows = [r for r in hard if r['dtype'] == dtype and r['method'] == method and int(r['classes']) == 16]
            y = np.abs(values(rows, 'grad_target'))
            ax.plot(values(rows, 'actual_margin'), np.where(y > 0, y, np.nan), color=COLORS[method], lw=1.6, ls='--' if method == 'upcast' else '-')
        ax.set(title=dtype, yscale='log', ylim=(1e-28, 1), xlim=(0, 60))
        ax.grid(alpha=0.15)
    for ax in axes[:, 0]: ax.set_ylabel('Absolute target gradient')
    for ax in axes[1]: ax.set_xlabel('Actual aggregate logit margin (nats)')
    fig.suptitle('Gradient disappearance | gaps in a line mean exactly zero', fontsize=13)
    legends(fig)
    save(fig, out / 'operator-target-gradient')


def cancellation_plot(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    for ax, dtype in zip(axes.flat, DTYPES):
        for method in COLORS:
            subset = [r for r in rows if r['dtype'] == dtype and r['method'] == method and float(r['requested_perturbation']) > 0]
            ax.plot(values(subset, 'requested_perturbation'), np.maximum(values(subset, 'gradient_error_l2'), 1e-30),
                    color=COLORS[method], lw=1.8, ls='--' if method == 'upcast' else '-')
        equal = [r for r in rows if r['dtype'] == dtype and r['method'] == 'native' and r['inputs_equal_after_quantization'] == 'True' and float(r['requested_perturbation']) > 0]
        if equal:
            ax.axvspan(1e-20, max(float(r['requested_perturbation']) for r in equal), color='#e9e9e9', alpha=0.65, zorder=-1)
        ax.set(title=dtype, xscale='log', yscale='log', xlim=(1e-20, 1), ylim=(1e-30, 1))
        ax.grid(alpha=0.15)
    for ax in axes[:, 0]: ax.set_ylabel('Gradient L2 error vs 80-digit reference')
    for ax in axes[1]: ax.set_xlabel('Requested perturbation eta, zS = zT + eta [-1, 1, 0]')
    fig.suptitle('Near-self-distillation error | shaded: inputs quantize to identical logits', fontsize=13)
    legends(fig)
    fig.text(0.5, -0.095, 'The reference uses the represented inputs. Zero error is displayed at 1e-30.', ha='center', fontsize=9)
    save(fig, out / 'cancellation-gradient-error')


def drift_plot(rows, out, dtypes, name):
    fig, axes = plt.subplots(2, len(dtypes), figsize=(6 * len(dtypes), 7), sharex=True, squeeze=False, constrained_layout=True)
    for col, dtype in enumerate(dtypes):
        for method in COLORS:
            subset = [r for r in rows if r['dtype'] == dtype and r['method'] == method and r['optimizer'] == 'adam']
            steps = sorted({int(r['optimizer_step']) for r in subset})
            for row_idx, key in enumerate(['parameter_drift_l2', 'canonical_kl']):
                ax = axes[row_idx, col]
                matrix = np.array([[float(r[key]) for r in subset if int(r['seed']) == seed] for seed in sorted({int(r['seed']) for r in subset})])
                ax.plot(steps, np.median(matrix, axis=0), color=COLORS[method], lw=1.8, ls='--' if method == 'upcast' else '-')
                ax.fill_between(steps, np.min(matrix, axis=0), np.max(matrix, axis=0), color=COLORS[method], alpha=0.12, linewidth=0)
                ax.set_yscale('symlog', linthresh=1e-8 if row_idx == 0 else 1e-14)
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.15)
        axes[0, col].set_title(dtype + (' | native/upcast coincide' if dtype in ('fp32', 'fp64') else ''))
        axes[1, col].set_xlabel('Optimizer step')
    axes[0, 0].set_ylabel('Parameter drift L2 from teacher')
    axes[1, 0].set_ylabel('KL of master models (before dtype cast)')
    fig.suptitle('Loss-boundary precision | fixed linear teacher and inputs | Adam, lr=0.002, eps=1e-8\nMedian and range of 3 seeds | 72 parameters | FP32 masters except FP64 arm', fontsize=12)
    legends(fig)
    save(fig, out / name)


def first_step_plot(rows, out):
    labels = [f'{d} / {m}' for d in DTYPES for m in COLORS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharey=True, constrained_layout=True)
    for ax, optimizer in zip(axes, ['sgd', 'adam']):
        for i, (dtype, method) in enumerate((d, m) for d in DTYPES for m in COLORS):
            vals = [float(r['actual_update_l2']) for r in rows if r['dtype'] == dtype and r['method'] == method and r['optimizer'] == optimizer]
            ax.scatter(vals, np.full(len(vals), i), color=COLORS[method], s=28, alpha=0.6)
            ax.scatter([np.median(vals)], [i], color=COLORS[method], marker='|', s=150)
        ax.set_xscale('symlog', linthresh=1e-15)
        ticks = [0, 1e-15, 1e-12, 1e-9, 1e-7] if optimizer == 'sgd' else [0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2]
        ax.set_xticks(ticks)
        ax.set_title(optimizer.upper() + ' | first actual update')
        ax.set_xlabel('L2 of stored parameter change (ideal = 0)')
        ax.grid(axis='x', alpha=0.2)
        ax.axvline(0, color='#555555', lw=0.8)
        ax.set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Does the residual gradient survive a real optimizer step? | 3 seeds', fontsize=13)
    save(fig, out / 'one-step-update')


def hard_update_plot(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
    for col, dtype in enumerate(['bf16', 'fp32']):
        for row_idx, op in enumerate(['sgd', 'adam']):
            ax = axes[row_idx, col]
            for method in COLORS:
                subset = [r for r in rows if r['dtype'] == dtype and r['method'] == method and int(r['classes']) == 16]
                ax.plot(values(subset, 'actual_margin'), np.maximum(values(subset, f'{op}_update_error_l2'), 1e-30),
                        color=COLORS[method], ls='--' if method == 'upcast' else '-', lw=1.8)
            ax.set(title=f'{dtype} | {op.upper()}', yscale='log', xlim=(0, 60))
            ax.grid(alpha=0.15)
    for ax in axes[:, 0]: ax.set_ylabel('Stored update L2 error vs reference-gradient step')
    for ax in axes[1]: ax.set_xlabel('Actual aggregate logit margin (nats)')
    fig.suptitle('Hard-target CE: identical parameters and fresh optimizer state | 16 classes', fontsize=13)
    legends(fig)
    fig.text(0.5, -0.095, 'Reference gradients use 80 digits, then the same master parameter dtype and torch optimizer. Zero error shown at 1e-30.', ha='center', fontsize=9)
    save(fig, out / 'hard-one-step-update-error')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    hard = load(args.results / 'operator.csv')
    cancellation = load(args.results / 'cancellation.csv')
    drift = load(args.results / 'distillation.csv')
    first = load(args.results / 'first_step.csv')
    operator_plot(hard, args.out)
    cancellation_plot(cancellation, args.out)
    drift_plot(drift, args.out, ['bf16', 'fp32'], 'self-distillation')
    drift_plot(drift, args.out, DTYPES, 'self-distillation-all-dtypes')
    first_step_plot(first, args.out)
    hard_update_plot(hard, args.out)
    print(f'Rendered 7 figure pairs to {args.out}')


if __name__ == '__main__':
    main()
