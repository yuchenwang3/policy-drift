#!/usr/bin/env python3
"""Analyze saved real-model results and render figures. Does not run the model."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from lab import dump_json, stable_log_softmax, write_csv


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def relative(actual, reference):
    denominator=np.linalg.norm(reference)
    return float(np.linalg.norm(actual-reference)/denominator) if denominator else None


def save(fig, path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--figures',type=Path,required=True)
    args=parser.parse_args()
    root,figures=args.results,args.figures
    figures.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    forward=np.load(root/'captured-forward.npz')
    z=torch.from_numpy(forward['logits']).double()
    targets=torch.from_numpy(forward['targets']).long()
    p=stable_log_softmax(z).exp()
    ref=p.clone()
    other=p.clone().scatter_(-1,targets[:,None],0)
    ref.scatter_(-1,targets[:,None],-other.sum(-1,keepdim=True))
    ref=(ref/len(targets)).numpy()
    baseline=np.load(root/'isolated-pt_fp32.npz')['after_cast_gradient']
    isolated=[]
    for method in ['pt_native','pt_fp32','stable_fp32','liger_native','liger_fp32']:
        data=np.load(root/f'isolated-{method}.npz')
        before=data['operand_gradient'].astype(np.float64)
        after=data['after_cast_gradient'].astype(np.float64)
        isolated.append({'method':method,'returned_operand_grad_relative_error':relative(before,ref),
                         'after_logit_cast_grad_relative_error':relative(after,ref),
                         'after_cast_different_coordinates_vs_pt_fp32':int(np.count_nonzero(after!=baseline)),
                         'explicit_cast_relative_error':relative(after,before),
                         'operand_dtype':next(r['operand_dtype'] for r in read_csv(root/'isolated-ce.csv') if r['method']==method)})
    write_csv(root/'ce-boundary-summary.csv',isolated)
    updates=summary['updates']
    erasure=[]
    for method in [h['method'] for h in summary['head']]:
        for optimizer in ['sgd','adam']:
            master=next(r for r in updates if r['method']==method and r['optimizer']==optimizer and r['writeback']=='fp32_master')
            low=next(r for r in updates if r['method']==method and r['optimizer']==optimizer and r['writeback']=='bf16_writeback')
            count=master['changed_parameters']
            erasure.append({'method':method,'optimizer':optimizer,'changed_fp32_parameters':count,
                            'changed_bf16_parameters':low['changed_parameters'],
                            'fraction_changed_updates_erased':1-low['changed_parameters']/count if count else 0.0,
                            'fp32_update_l2':master['update_norm'],'bf16_update_l2':low['update_norm'],
                            'fp32_policy_kl':master['policy_kl_mean'],'bf16_policy_kl':low['policy_kl_mean']})
    write_csv(root/'writeback-erasure.csv',erasure)
    controlled=[]
    for method in ['ctrl_logits_round','ctrl_hidden_round','ctrl_weight_round']:
        for optimizer in ['sgd','adam']:
            for writeback in ['fp32_master','bf16_writeback']:
                a=np.load(root/f'samples-{method}-{optimizer}-{writeback}.npz')
                b=np.load(root/f'samples-ctrl_keep-{optimizer}-{writeback}.npz')
                keys=[key for key in a.files if key.endswith(':after')]
                diff=np.concatenate([(a[k].astype(np.float64)-b[k].astype(np.float64)).ravel() for k in keys])
                before_equal=all(np.array_equal(a[k.replace(':after',':before')],b[k.replace(':after',':before')]) for k in keys)
                controlled.append({'method':method,'optimizer':optimizer,'writeback':writeback,
                                   'reference':'ctrl_keep','sampled_coordinates':len(diff),
                                   'sampled_parameter_blocks':len(keys),'same_sampled_initial_parameters':before_equal,
                                   'different_sampled_after_step_coordinates':int(np.count_nonzero(diff)),
                                   'sampled_update_difference_l2':float(np.linalg.norm(diff)),
                                   'sampled_update_difference_abs_max':float(np.abs(diff).max())})
    write_csv(root/'controlled-sample-differences.csv',controlled)
    findings={'model':summary['model'],'revision':summary['revision'],'parameter_count':manifest['parameter_count'],
              'prefix_count':manifest['prefix_count'],'actual_optimizer_steps':summary['actual_optimizer_steps'],
              'gradient_methods':summary['gradient_methods'],'update_evaluation_rows':summary['update_evaluation_rows'],
              'self_stable_exact_zero_updates':summary['self_stable_exact_zero_updates'],
              'isolated_ce':isolated,'gradient_methods_table':summary['head'],
              'writeback_erasure':erasure,'controlled_samples':controlled,'updates':updates,
              'control_cast_traces':summary['cast_controls'],
              'initial_probabilities':summary['initial_probabilities'],
              'scope':{'hard_gradient_reference':'PT FP32 CE on the shared BF16 logits, same upstream graph',
                       'isolated_reference':'stable FP64 gradient on exactly the captured BF16 logits',
                       'self_reference':'zero gradient of the ideal logit-defined same-model soft-target objective',
                       'flce_logits':'internal logits are not exported; the recorded logits are an external batched capture',
                       'cast_controls':'diagnostic FP32 head VJPs with exactly one additional BF16 cast',
                       'control_update_differences':'1024 saved coordinates in four parameter blocks; not whole-model pairwise norms',
                       'writeback':'projection of the completed FP32 optimizer step to BF16; not BF16 optimizer-state arithmetic'}}
    dump_json(root/'findings.json',findings)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    x=np.arange(len(isolated)); width=.36
    axes[0].bar(x-width/2,[r['returned_operand_grad_relative_error'] for r in isolated],width,color='#417ba0',label='Returned loss-operand gradient')
    axes[0].bar(x+width/2,[r['after_logit_cast_grad_relative_error'] for r in isolated],width,color='#c76b31',label='After BF16 logit boundary')
    axes[0].set_xticks(x,['PT\nBF16','PT\nFP32','Stable\nFP32','Liger\nBF16','Liger\nFP32'])
    axes[0].set(yscale='log',ylabel='Relative gradient L2 error vs stable FP64',title='Identical captured logits | full vocabulary')
    axes[0].legend(fontsize=8,loc='lower left')
    axes[0].grid(axis='y',alpha=.15)
    locations=[('ctrl_logits_round','logits_gradient','dL/dlogits'),
               ('ctrl_hidden_round','hidden_gradient','dL/dhidden'),
               ('ctrl_weight_round','head_weight_gradient','dL/dhead weight')]
    traces=summary['cast_controls']
    axes[1].bar(np.arange(3),[traces[a][b]['relative_l2_error'] for a,b,c in locations],color='#77549a')
    axes[1].set_xticks(np.arange(3),[c for a,b,c in locations])
    axes[1].set(ylabel='Local relative L2 error introduced by one cast',title='Same forward | one BF16 backward cast at a time')
    axes[1].grid(axis='y',alpha=.15)
    fig.suptitle('Qwen2.5-0.5B | fixed 8 prefixes | BF16 AMP, FP32 master parameters',fontsize=12)
    save(fig,figures/'real-model-ce-boundaries')
    hard=[h['method'] for h in summary['head'] if h['objective']=='hard_next_token']
    labels=['PT FP32','PT BF16','Stable FP32','Liger BF16','Liger FP32',
            'PT FP32, chunk=1','Liger fused linear CE','Control: keep FP32',
            'Control: round dlogits','Control: round dhidden','Control: round dW']
    fig,axes=plt.subplots(1,2,figsize=(13,6.3),sharey=True,constrained_layout=True)
    for optimizer,color,offset in [('sgd','#417ba0',-.12),('adam','#c76b31',.12)]:
        left=[next(r for r in updates if r['method']==m and r['optimizer']==optimizer and r['writeback']=='fp32_master') for m in hard]
        right=[next(r for r in erasure if r['method']==m and r['optimizer']==optimizer) for m in hard]
        axes[0].scatter([r['update_relative_error_vs_reference'] for r in left],np.arange(len(hard))+offset,color=color,label=optimizer.upper(),s=32)
        axes[1].scatter([100*r['fraction_changed_updates_erased'] for r in right],np.arange(len(hard))+offset,color=color,label=optimizer.upper(),s=32)
    axes[0].set_xscale('symlog',linthresh=1e-8)
    axes[0].set_xticks([0,1e-7,1e-5,1e-3,1e-1])
    axes[0].set(xlabel='Relative update L2 difference vs PT FP32',title='Actual FP32 optimizer step | fresh state, lr=1e-5')
    axes[1].set(xlabel='Changed FP32 coordinates erased by BF16 writeback (%)',xlim=(-2,102),title='Same completed step, then cast weights to BF16')
    axes[0].set_yticks(np.arange(len(hard)),labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis='x',alpha=.2)
        ax.legend(loc='lower right',fontsize=9)
    fig.suptitle('Real-model gradients, one-step updates and final writeback have different equivalence boundaries',fontsize=12)
    save(fig,figures/'real-model-one-step')
    soft=['self_pt_native','self_pt_fp32','self_stable_fp32']
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    grad=[next(h for h in summary['head'] if h['method']==m)['parameter_gradient_norm'] for m in soft]
    axes[0].scatter(np.arange(3),grad,color='#596a79',s=50)
    axes[0].set(yscale='symlog',ylabel='Full-model parameter gradient L2',title='Same teacher/student logits')
    axes[0].set_yscale('symlog',linthresh=1e-9)
    for op,color,offset in [('sgd','#417ba0',-.08),('adam','#c76b31',.08)]:
        rows=[next(r for r in updates if r['method']==m and r['optimizer']==op and r['writeback']=='fp32_master') for m in soft]
        axes[1].scatter(np.arange(3)+offset,[r['update_norm'] for r in rows],color=color,label=op.upper(),s=45)
        axes[2].scatter(np.arange(3)+offset,[r['policy_kl_mean'] for r in rows],color=color,label=op.upper(),s=45)
    axes[1].set(title='Actual first stored update',ylabel='Parameter update L2')
    axes[1].set_yscale('symlog',linthresh=1e-11)
    axes[2].set(title='Same-prefix executed-policy change',ylabel='Mean KL(old || after one step)')
    axes[2].set_yscale('symlog',linthresh=1e-8)
    for ax in axes:
        ax.set_xticks(np.arange(3),['Native\nBF16','Native\nFP32','Logit-aware\nstable FP32'])
        ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.15)
    axes[2].legend(fontsize=9)
    fig.suptitle('Self-distillation | fixed 8 prefixes, T=4 | lr=1e-5, Adam eps=1e-8 | no weight decay',fontsize=12)
    save(fig,figures/'real-model-self-distillation')
    print(json.dumps({'status':'complete','figures':3,'gradient_methods':len(summary['head']),
                      'actual_optimizer_steps':summary['actual_optimizer_steps']}))


if __name__=='__main__':
    main()
