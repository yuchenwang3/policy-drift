#!/usr/bin/env python3
"""Factorial ablation of Adam m, v and step on the two saved minimal cases.

Parameters stay at the original checkpoint for every probe. Each configuration
has an explicit zero-gradient paired control. Only missing complete Adam states
are rebuilt; existing fresh-state results are reused. No RL or generation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab import dump_json, stable_log_softmax, write_csv
from real_model_step import digest_file, restore
from minimize_self_distillation import Experiment, update_metrics, vector_norm, weights


def safe_details(before, after, tokenizer):
    finite = bool(torch.isfinite(before).all() and torch.isfinite(after).all())
    if not finite:
        return {'finite_logits': False, 'nonfinite_before': int(torch.count_nonzero(~torch.isfinite(before))),
                'nonfinite_after': int(torch.count_nonzero(~torch.isfinite(after))),
                'kl_mean': None, 'kl_max': None, 'different_logits': None,
                'argmax_different_prefixes': None, 'before_argmax_ids': None,
                'after_argmax_ids': None, 'before_argmax_text': None, 'after_argmax_text': None}
    a, b = before.double(), after.double()
    la, lb = stable_log_softmax(a), stable_log_softmax(b)
    p = la.exp()
    d = b-a
    d = d-(p*d).sum(-1,keepdim=True)
    if float(d.abs().max()) < 250:
        series=d.square()*(0.5+d*(1/6+d*(1/24+d*(1/120+d/720))))
        remainder=torch.where(d.abs()<0.01,series,torch.expm1(d)-d)
        kl=torch.log1p((p*remainder).sum(-1))
        calculation='centered expm1/log1p, small-delta series'
    else:
        # Removing v while retaining m can create enormous counterfactual steps.
        # Log probabilities avoid overflow from expm1 in that regime.
        kl=(p*(la-lb)).sum(-1)
        calculation='direct FP64 log-probability KL for large displacement'
    old_ids,new_ids=a.argmax(-1).tolist(),b.argmax(-1).tolist()
    return {'finite_logits': True, 'kl_mean': float(kl.mean()), 'kl_max': float(kl.max()),
            'different_logits': int(torch.count_nonzero(a!=b)),
            'argmax_different_prefixes': sum(x!=y for x,y in zip(old_ids,new_ids)),
            'before_argmax_ids': old_ids, 'after_argmax_ids': new_ids,
            'before_argmax_text': [tokenizer.decode([x]) for x in old_ids],
            'after_argmax_text': [tokenizer.decode([x]) for x in new_ids],
            'kl_calculation': calculation}


def finite_update(parameters, baseline):
    metric=update_metrics(parameters,baseline)
    metric['finite_update_norm']=math.isfinite(metric['update_l2'])
    if not metric['finite_update_norm']: metric['update_l2']=None
    return metric


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value


def transformed_state(snapshot, keep_m, keep_v, keep_step):
    result=copy.deepcopy(snapshot)
    for entry in result['state'].values():
        if not keep_m: entry['exp_avg'].zero_()
        if not keep_v: entry['exp_avg_sq'].zero_()
        if not keep_step: entry['step'].zero_()
    return result


class Ablation(Experiment):
    def __init__(self,*args,prior,prior_dir,**kwargs):
        super().__init__(*args,**kwargs)
        self.prior=prior
        self.prior_dir=prior_dir
        self.cached={}
        self.results=[]
        self.state_records=[]
        self.diagnostics=[]

    def cache_residuals(self,cases):
        for label,case in cases.items():
            restore(self.parameters,self.initial)
            before,loss,norm=self.residual(case)
            gradients={name:p.grad.detach().clone() for name,p in self.parameters.items() if p.grad is not None}
            old=next(r for r in self.prior['moment_comparisons'] if r['label']==label+'-initial-fresh')
            old_forward=np.load(self.prior_dir/f'paired-{label}-initial-fresh.npz')['before']
            forward_matches=bool(np.array_equal(before.float().cpu().numpy(),old_forward))
            self.cached[label]={'before':before,'loss':loss,'norm':norm,'gradients':gradients,
                                'norm_matches_prior':norm==old['native_gradient_branch']['gradient_l2'],
                                'forward_matches_prior':forward_matches}
            dump_json(self.out/f'residual-cache-{label}.json',
                      {'gradient_l2':norm,'loss':loss,'parameter_tensors_with_gradient':len(gradients),
                       'norm_matches_prior':self.cached[label]['norm_matches_prior'],
                       'forward_matches_prior':forward_matches})
            self.model.zero_grad(set_to_none=True)
        restore(self.parameters,self.initial)

    def diagnostics_for_state(self,opt,step):
        for label,cached in self.cached.items():
            counts={'nonzero_residual_coordinates':0,'nonzero_residual_below_epsilon':0,
                    'history_v_dominates_new_v_coordinates':0}
            energy=history_energy=0.0
            for name,p in self.parameters.items():
                if name not in cached['gradients']: continue
                g=cached['gradients'][name].double()
                v=opt.state[p]['exp_avg_sq'].double()
                nonzero=g!=0
                dominant=0.999*v > 0.001*g.square()
                counts['nonzero_residual_coordinates']+=int(torch.count_nonzero(nonzero))
                counts['nonzero_residual_below_epsilon']+=int(torch.count_nonzero(nonzero & (g.abs()<1e-8)))
                counts['history_v_dominates_new_v_coordinates']+=int(torch.count_nonzero(nonzero & dominant))
                energy+=float(g.square().sum())
                history_energy+=float(torch.where(dominant,g.square(),0).sum())
            row={'case_label':label,'warmup_step':step,**counts,
                 'history_v_dominates_residual_gradient_energy_fraction':history_energy/energy if energy else None}
            self.diagnostics.append(row)
        dump_json(self.out/'denominator-diagnostics.json',self.diagnostics)

    def save_state(self,opt,current,step,state_dir,prepared,prior_dir):
        info,samples=self.state_summary(opt)
        old=json.loads((prior_dir/f'adam-state-step-{step}.json').read_text())
        comparisons={key:info[key]==old[key] for key in info}
        np.savez_compressed(self.out/f'rebuilt-state-samples-{step}.npz',**samples)
        state_dir.mkdir(parents=True,exist_ok=True)
        target=state_dir/f'checkpoint-{step}.pt'
        if target.exists(): raise RuntimeError('Complete state already exists; use --load-states instead of rebuilding it')
        temp=target.with_suffix('.partial')
        payload={'parameters':cpu_tree(current),'optimizer_state_dict':cpu_tree(opt.state_dict()),
                 'model':prepared['model'],'revision':prepared['revision'],'warmup_steps':step,
                 'parameter_order':list(self.parameters),'state_summary':info,
                 'source_sha256':digest_file(__file__)}
        torch.save(payload,temp)
        os.chmod(temp,0o600)
        temp.replace(target)
        record={'warmup_step':step,'path':str(target),'bytes':target.stat().st_size,
                'sha256':digest_file(target),'statistics':info,
                'prior_statistics_and_samples_match':comparisons,
                'prior_full_state_bitwise_comparison_available':False,
                'contents':'complete warmup parameter tensors and Adam optimizer state'}
        self.state_records.append(record)
        dump_json(self.out/'state-cache-receipt.json',self.state_records)
        print(json.dumps({'phase':'state-saved','step':step,'path':str(target),
                          'bytes':record['bytes'],'prior_statistics_match':all(comparisons.values())}),flush=True)
        del payload
        return record

    def reused_fresh(self,label,case,step):
        source=next(r for r in self.prior['moment_comparisons'] if r['label']==label+'-initial-fresh')
        row=copy.deepcopy(source)
        row.update({'case_label':label,'warmup_step':step,'keep_m':False,'keep_v':False,'keep_step':False,
                    'configuration':'m0-v0-t0','execution':'reused-existing',
                    'source_comparison':source['label'],'source_results':str(self.prior_dir.resolve()),
                    'state_kind':'factorial-counterfactual','case':case})
        self.results.append(row)

    def pair_ablation(self,label,case,step,snapshot,flags):
        km,kv,kt=flags
        name=f'm{int(km)}-v{int(kv)}-t{int(kt)}'
        state=transformed_state(snapshot,*flags)
        cached=self.cached[label]
        before=cached['before']
        branches={}
        zero_parameters=zero_logits=None
        for kind in ['zero','native']:
            restore(self.parameters,self.initial)
            self.model.zero_grad(set_to_none=True)
            opt=self.optimizer(state)
            for parameter_name,p in self.parameters.items():
                p.grad=(cached['gradients'][parameter_name].clone()
                        if kind=='native' and parameter_name in cached['gradients'] else torch.zeros_like(p))
            opt.step()
            self.log_step('state-ablation',case_label=label,warmup_state_step=step,
                          configuration=name,gradient_kind=kind,
                          gradient_norm=cached['norm'] if kind=='native' else 0.0)
            after=self.forward(case)
            branches[kind]={'gradient_l2':cached['norm'] if kind=='native' else 0.0,
                            'loss':cached['loss'] if kind=='native' else None,
                            **finite_update(self.parameters,self.initial),
                            **safe_details(before,after,self.tokenizer)}
            if kind=='zero':
                zero_parameters=weights(self.parameters)
                zero_logits=after.clone()
            else:
                delta=finite_update(self.parameters,zero_parameters)
                policy=safe_details(zero_logits,after,self.tokenizer)
                np.savez_compressed(self.out/f'paired-{label}-state{step}-{name}.npz',
                                    before=before.float().cpu().numpy(),
                                    zero_after=zero_logits.float().cpu().numpy(),native_after=after.float().cpu().numpy())
            del opt
            self.model.zero_grad(set_to_none=True)
        row={'case_label':label,'warmup_step':step,'keep_m':km,'keep_v':kv,'keep_step':kt,
             'configuration':name,'execution':'new-paired-probe',
             'adam_step_before':step if kt else 0,'adam_step_used':step+1 if kt else 1,
             'zero_gradient_control':branches['zero'],'native_gradient_branch':branches['native'],
             'native_minus_zero_update_l2':delta['update_l2'],
             'native_minus_zero_changed_parameter_coordinates':delta['changed_parameter_coordinates'],
             'native_vs_zero_policy':policy,'case':case}
        self.results.append(row)
        dump_json(self.out/'ablation-results.json',self.results)
        print(json.dumps({'phase':'ablation','case':label,'state_step':step,'configuration':name,
                          'extra_update_l2':delta['update_l2'],'extra_kl':policy['kl_mean'],
                          'extra_argmax_changes':policy['argmax_different_prefixes'],
                          'finite_logits':policy['finite_logits']}),flush=True)
        del state,zero_parameters,zero_logits,after

    def stage(self,cases,snapshot,step):
        for label,case in cases.items():
            cached=self.cached[label]
            if cached['norm_matches_prior'] and cached['forward_matches_prior']:
                self.reused_fresh(label,case,step)
            else:
                self.pair_ablation(label,case,step,snapshot,(False,False,False))
            for flags in itertools.product([False,True],repeat=3):
                if not any(flags): continue
                self.pair_ablation(label,case,step,snapshot,flags)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--original-results',type=Path,required=True)
    parser.add_argument('--prior-results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--state-cache-dir',type=Path)
    parser.add_argument('--load-states',type=Path)
    parser.add_argument('--run-id', default=None, help='Explicit W&B run ID; omit for a fresh run.')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='offline')
    parser.add_argument('--wandb-project', default='policy-drift')
    parser.add_argument('--wandb-entity', default=None)
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): parser.error('Existing experiment retained; choose a new output directory')
    prepared=json.loads(args.prepared.read_text())
    prior=json.loads((args.prior_results/'summary.json').read_text())
    original=json.loads((args.original_results/'prefixes.json').read_text())
    cases=prior['probe_cases']
    state_dir=(args.state_cache_dir or out.parent/'states').resolve()
    if not args.load_states and any((state_dir/f'checkpoint-{s}.pt').exists() for s in [16,128]):
        parser.error('Saved complete states exist; pass --load-states to reuse them')
    torch.manual_seed(0);torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(prepared['snapshot'],dtype=torch.bfloat16,
                attn_implementation='eager',local_files_only=True,trust_remote_code=False).float().cuda().eval()
    model.config.use_cache=False
    tokenizer=AutoTokenizer.from_pretrained(prepared['snapshot'],local_files_only=True,trust_remote_code=False)
    manifest={'status':'running','host':socket.gethostname(),'pid':os.getpid(),
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'model':prepared['model'],'revision':prepared['revision'],'gpu':torch.cuda.get_device_name(),
              'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,
              'source_sha256':digest_file(__file__),
              'reused_minimum_sha256':digest_file(Path(__file__).with_name('minimize_self_distillation.py')),
              'reused_real_model_sha256':digest_file(Path(__file__).with_name('real_model_step.py')),
              'reused_lab_sha256':digest_file(Path(__file__).with_name('lab.py')),
              'learning_rate':1e-5,'epsilon':1e-8,'betas':[0.9,0.999],'weight_decay':0,
              'temperature':4,'temperature_squared_scaling':False,
              'probe_parameters':'original fixed FP32 parameters, BF16 AMP forward',
              'factorial':'m keep/zero x v keep/zero x pre-step count keep/zero',
              'state_steps':[16,128],'state_source':'loaded-complete-cache' if args.load_states else 'rebuilt-original-128-step-history',
              'prior_results':str(args.prior_results.resolve()),'state_cache_dir':str(state_dir),
              'scope':'paired state-component ablation; synthetic combinations; no RL or generation'}
    dump_json(out/'manifest.json',manifest)
    tracker = None
    if args.wandb_mode != 'disabled':
        import wandb
        tracker=wandb.init(project=args.wandb_project, entity=args.wandb_entity,id=args.run_id,
                    name='fixed-weights-adam-m-v-step-factorial',mode=args.wandb_mode,dir=str(out),config=manifest,
                    tags=['adam-ablation','fixed-weights','minimal-counterexample'])
        tracker.log({'optimizer_step':0},step=0)
    exp=Ablation(model,tokenizer,out,tracker,prior=prior,prior_dir=args.prior_results)
    exp.cache_residuals(cases)
    if args.load_states:
        for step in [16,128]:
            path=args.load_states/f'checkpoint-{step}.pt'
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if payload['model']!=prepared['model'] or payload['revision']!=prepared['revision']:
                raise RuntimeError('State cache belongs to another model revision')
            if payload['parameter_order']!=list(exp.parameters): raise RuntimeError('Parameter ordering differs')
            exp.stage(cases,payload['optimizer_state_dict'],step)
            exp.state_records.append({'warmup_step':step,'path':str(path.resolve()),'bytes':path.stat().st_size,
                                      'sha256':digest_file(path),'loaded_existing':True})
            del payload
        warmup_steps=0
    else:
        warm_case={'rows':[{'origin':i,'family':r['family'],'input_ids':r['input_ids']} for i,r in enumerate(original['prefixes'])],
                   'width':len(original['input_ids'][0]),'pad_id':cases['original-transition']['pad_id'],'target_origin':None}
        targets=torch.tensor(original['target_ids'],device='cuda',dtype=torch.long)
        restore(exp.parameters,exp.initial)
        opt=exp.optimizer()
        for step in range(1,129):
            model.zero_grad(set_to_none=True)
            z=exp.forward(warm_case,grad=True)
            with torch.autocast('cuda',enabled=False): loss=F.cross_entropy(z.float(),targets)
            loss.backward()
            g=vector_norm(p.grad for p in exp.parameters.values() if p.grad is not None)
            opt.step()
            exp.log_step('rebuild-supervised-history',loss=float(loss.detach()),gradient_norm=g,warmup_optimizer_step=step)
            exp.warmup.append({'step':step,'loss':float(loss.detach()),'gradient_l2':g,'learning_rate':exp.lr})
            if step in [16,128]:
                current=weights(exp.parameters)
                exp.save_state(opt,current,step,state_dir,prepared,args.prior_results)
                exp.diagnostics_for_state(opt,step)
                snapshot=copy.deepcopy(opt.state_dict())
                exp.stage(cases,snapshot,step)
                restore(exp.parameters,current)
                del current,snapshot
                write_csv(out/'warmup.csv',exp.warmup)
        warmup_steps=128
    coefficients=[]
    for previous_step in [0,16,128]:
        t=previous_step+1
        a=0.1/(1-0.9**t);b=math.sqrt(0.001/(1-0.999**t))
        coefficients.append({'pre_step_count':previous_step,'update_step':t,'numerator_coefficient_a':a,
                             'absolute_gradient_denominator_coefficient_b':b,
                             'effective_epsilon_after_factoring_b':1e-8/b,
                             'large_gradient_step_multiplier_vs_lr':a/b,
                             'small_gradient_slope_multiplier_vs_fresh':a})
    summary={'status':'complete','actual_optimizer_steps':exp.steps,'warmup_optimizer_steps':warmup_steps,
             'new_paired_probe_steps':exp.steps-warmup_steps,'factorial_rows':len(exp.results),
             'new_factorial_rows':sum(r['execution']=='new-paired-probe' for r in exp.results),
             'reused_factorial_rows':sum(r['execution']=='reused-existing' for r in exp.results),
             'unique_reused_fresh_comparisons':2,'cases':cases,'results':exp.results,
             'state_cache':exp.state_records,'denominator_diagnostics':exp.diagnostics,
             'zero_m_zero_v_analytic_coefficients':coefficients,
             'scope':'original weights only; component-reset counterfactual states; next-token BF16 executed policy; no RL'}
    dump_json(out/'summary.json',summary)
    if tracker is not None:
        tracker.summary.update({'status':'complete','actual_optimizer_steps':exp.steps,
                                'warmup_steps':warmup_steps,'factorial_rows':len(exp.results)})
        tracker.finish()
    manifest.update({'status':'complete','actual_optimizer_steps':exp.steps,
                     'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    dump_json(out/'manifest.json',manifest)
    print(json.dumps({'status':'complete','actual_optimizer_steps':exp.steps,'factorial_rows':len(exp.results)}),flush=True)


if __name__=='__main__': main()
