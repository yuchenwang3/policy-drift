#!/usr/bin/env python3
"""Read saved state-ablation outputs and expose update and decision margins."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def margin(logits):
    z=logits[0].astype(np.float64)
    if not np.isfinite(z).all(): return None
    top=np.partition(z,-2)[-2:]
    return float(top.max()-top.min())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--prior-results',type=Path,required=True)
    parser.add_argument('--publish-dir',type=Path)
    args=parser.parse_args()
    s=json.loads((args.results/'summary.json').read_text())
    rows=[]
    for r in s['results']:
        label,step,config=r['case_label'],r['warmup_step'],r['configuration']
        if r['execution']=='reused-existing':
            path=args.prior_results/f'paired-{label}-initial-fresh.npz'
        else:
            path=args.results/f'paired-{label}-state{step}-{config}.npz'
        data=np.load(path)
        zero,native=data['zero_after'],data['native_after']
        policy=r['native_vs_zero_policy']
        nz=r['native_gradient_branch']; zz=r['zero_gradient_control']
        zero_id=int(zero[0].argmax()) if np.isfinite(zero).all() else None
        native_id=int(native[0].argmax()) if np.isfinite(native).all() else None
        rows.append({'case':label,'state_step':step,'configuration':config,
                     'keep_m':r['keep_m'],'keep_v':r['keep_v'],'keep_step':r['keep_step'],
                     'execution':r['execution'],'residual_gradient_l2':nz['gradient_l2'],
                     'zero_branch_update_l2':zz['update_l2'],
                     'residual_extra_update_l2':r['native_minus_zero_update_l2'],
                     'residual_extra_policy_kl':policy['kl_mean'],
                     'residual_extra_argmax_changes':policy['argmax_different_prefixes'],
                     'zero_branch_argmax_id':zero_id,'native_branch_argmax_id':native_id,
                     'zero_branch_token':zz['after_argmax_text'][0] if zz['after_argmax_text'] else None,
                     'native_branch_token':nz['after_argmax_text'][0] if nz['after_argmax_text'] else None,
                     'zero_branch_top2_logit_gap':margin(zero),
                     'native_branch_top2_logit_gap':margin(native),
                     'zero_branch_gap_to_native_winner':float(zero[0].max()-zero[0,native_id]) if native_id is not None and zero_id is not None else None,
                     'finite_logits':policy.get('finite_logits',True)})
    import csv
    with (args.results/'ablation-summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'status':'complete','factorial_rows':len(rows),'actual_optimizer_steps':s['actual_optimizer_steps'],
             'warmup_optimizer_steps':s['warmup_optimizer_steps'],'new_probe_optimizer_steps':s['new_paired_probe_steps'],
             'new_rows':s['new_factorial_rows'],'reused_rows':s['reused_factorial_rows'],
             'results':rows,'state_cache':s['state_cache'],
             'denominator_diagnostics':s['denominator_diagnostics'],
             'zero_m_zero_v_coefficients':s['zero_m_zero_v_analytic_coefficients'],
             'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'scope':'fixed original weights; paired residual-minus-zero; counterfactual Adam component resets; no RL'}
    (args.results/'mechanisms.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if args.publish_dir:
        args.publish_dir.mkdir(parents=True, exist_ok=True)
        for name in ['ablation-summary.csv','mechanisms.json']:
            shutil.copyfile(args.results/name,args.publish_dir/name)
    print(json.dumps({'status':'complete','rows':len(rows),'actual_optimizer_steps':s['actual_optimizer_steps']}))


if __name__=='__main__': main()
