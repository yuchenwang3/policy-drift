#!/usr/bin/env python3
"""Fixed-original-weight self-distillation probes with Muon + AdamW.

Uses the two existing minimal inputs and the existing eight-example warmup.
Group endpoint mixing isolates residual effects without extra optimizer steps.
"""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab import dump_json
from real_model_step import digest_file, restore
from minimize_self_distillation import Experiment, vector_norm, weights
from ablate_adam_state import safe_details, cpu_tree


def groups_for(model, parameters):
    excluded={id(model.get_input_embeddings().weight), id(model.lm_head.weight)}
    muon=[n for n,p in parameters.items() if p.ndim==2 and id(p) not in excluded]
    adamw=[n for n in parameters if n not in set(muon)]
    return {'muon':muon,'adamw':adamw}


def optimizers(parameters,groups,state=None):
    result={
        'muon':torch.optim.Muon([parameters[n] for n in groups['muon']],lr=1e-5,
                    weight_decay=0,momentum=0.95,nesterov=True,eps=1e-7,ns_steps=5,
                    adjust_lr_fn='match_rms_adamw'),
        'adamw':torch.optim.AdamW([parameters[n] for n in groups['adamw']],lr=1e-5,
                    betas=(0.9,0.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)}
    if state is not None:
        for group,opt in result.items(): opt.load_state_dict(copy.deepcopy(state[group]))
    return result


def snapshot_optimizers(opts):
    return {group:copy.deepcopy(opt.state_dict()) for group,opt in opts.items()}


@torch.no_grad()
def group_updates(parameters,reference,groups):
    result={}
    for group,names in groups.items():
        squared=0.0;changed=0
        for name in names:
            delta=parameters[name].detach().double()-reference[name].double()
            squared+=float(delta.square().sum())
            changed+=int(torch.count_nonzero(delta))
        result[group]={'update_l2':squared**0.5,'changed_coordinates':changed}
    return result


def state_statistics(opts):
    output={}
    for group,opt in opts.items():
        fields={}
        for key in ['momentum_buffer','exp_avg','exp_avg_sq']:
            tensors=[s[key] for s in opt.state.values() if key in s]
            if tensors:
                fields[key]={'l2':vector_norm(tensors),
                             'nonzero_coordinates':sum(int(torch.count_nonzero(t)) for t in tensors)}
        fields['step_counts']=sorted({int(s['step']) for s in opt.state.values() if 'step' in s})
        fields['parameter_states']=len(opt.state)
        output[group]=fields
    return output


def probe(exp,cases,cached,groups,state,state_step,results):
    for label,case in cases.items():
        data=cached[label];before=data['before'];branches={}
        for kind in ['zero','native']:
            restore(exp.parameters,exp.initial)
            exp.model.zero_grad(set_to_none=True)
            opts=optimizers(exp.parameters,groups,state)
            for name,p in exp.parameters.items():
                p.grad=data['gradients'][name].clone() if kind=='native' else torch.zeros_like(p)
            for opt in opts.values(): opt.step()
            exp.log_step('hybrid-probe',warmup_state_step=state_step,case_label=label,
                         gradient_kind=kind,gradient_norm=data['norm'] if kind=='native' else 0.0)
            after=exp.forward(case)
            branches[kind]={'group_updates':group_updates(exp.parameters,exp.initial,groups),
                            'policy':safe_details(before,after,exp.tokenizer)}
            if kind=='zero':
                zero_parameters=weights(exp.parameters);zero_logits=after.clone()
            else:
                native_parameters=weights(exp.parameters);native_logits=after.clone()
                extra=group_updates(exp.parameters,zero_parameters,groups)
            del opts
            exp.model.zero_grad(set_to_none=True)
        isolated={};arrays={}
        for selected,names in groups.items():
            selected_names=set(names)
            with torch.no_grad():
                for name,p in exp.parameters.items():
                    p.copy_(native_parameters[name] if name in selected_names else zero_parameters[name])
            z=exp.forward(case)
            isolated[selected]=safe_details(zero_logits,z,exp.tokenizer)
            arrays['only_'+selected]=z.float().cpu().numpy()
        row={'case':label,'state_step':state_step,'zero_gradient_control':branches['zero'],
             'native_gradient_branch':branches['native'],'group_residual_updates':extra,
             'joint_residual_policy':safe_details(zero_logits,native_logits,exp.tokenizer),
             'isolated_group_residual_policy':isolated,'group_gradient_l2':data['group_norms'],
             'case_fixture':case}
        results.append(row)
        np.savez_compressed(exp.out/f'paired-{label}-state-{state_step}.npz',
                 before=before.float().cpu().numpy(),zero_after=zero_logits.float().cpu().numpy(),
                 native_after=native_logits.float().cpu().numpy(),**arrays)
        dump_json(exp.out/'muon-comparisons.json',results)
        print(json.dumps({'case':label,'state_step':state_step,
            'extra_updates':{g:x['update_l2'] for g,x in extra.items()},
            'joint_extra_kl':row['joint_residual_policy']['kl_mean'],
            'joint_extra_argmax':row['joint_residual_policy']['argmax_different_prefixes'],
            'isolated_argmax':{g:x['argmax_different_prefixes'] for g,x in isolated.items()}}),flush=True)
        del zero_parameters,native_parameters,zero_logits,native_logits,after,z
        restore(exp.parameters,exp.initial)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--original-results',type=Path,required=True)
    parser.add_argument('--prior-results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--state-cache-dir',type=Path,required=True)
    parser.add_argument('--publish-dir',type=Path)
    parser.add_argument('--run-id', default=None, help='Explicit W&B run ID; omit for a fresh run.')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='offline')
    parser.add_argument('--wandb-project', default='numerical-policy-lab')
    parser.add_argument('--wandb-entity', default=None)
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): parser.error('Existing run retained; choose a new output directory')
    if any((args.state_cache_dir/f'muon-checkpoint-{s}.pt').exists() for s in [16,128]):
        parser.error('Existing hybrid states retained; do not repeat their history')
    prepared=json.loads(args.prepared.read_text())
    prior=json.loads((args.prior_results/'summary.json').read_text())
    original=json.loads((args.original_results/'prefixes.json').read_text())
    cases=prior['probe_cases']
    torch.manual_seed(0);torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(prepared['snapshot'],dtype=torch.bfloat16,
              attn_implementation='eager',local_files_only=True,trust_remote_code=False).float().cuda().eval()
    model.config.use_cache=False
    tokenizer=AutoTokenizer.from_pretrained(prepared['snapshot'],local_files_only=True,trust_remote_code=False)
    parameters=dict(model.named_parameters());groups=groups_for(model,parameters)
    group_info={g:{'parameter_tensors':len(names),'parameter_count':sum(parameters[n].numel() for n in names),
                   'names':names} for g,names in groups.items()}
    kernel=Path(importlib.import_module('torch.optim._muon').__file__)
    manifest={'status':'running','host':socket.gethostname(),'pid':os.getpid(),
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'command':sys.argv,'model':prepared['model'],'revision':prepared['revision'],
              'torch':torch.__version__,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'source_sha256':digest_file(__file__),'muon_kernel_source_sha256':digest_file(kernel),
              'muon_kernel_source_path':str(kernel),'group_info':group_info,
              'tied_embedding_head':model.lm_head.weight is model.get_input_embeddings().weight,
              'muon':{'lr':1e-5,'momentum':0.95,'nesterov':True,'eps':1e-7,'ns_steps':5,
                      'adjust_lr_fn':'match_rms_adamw','weight_decay':0},
              'adamw':{'lr':1e-5,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0},
              'precision':'FP32 master, BF16 AMP forward; installed Muon NS uses BF16',
              'temperature':4,'temperature_squared_scaling':False,
              'warmup':'actual Muon+AdamW hard-label updates on original eight fixtures',
              'probe':'original fixed weights; same hybrid state; residual versus explicit zero gradient',
              'group_isolation':'mix final group endpoints with the other group from the zero-gradient branch',
              'scope':'one selected diagnostic configuration; no RL, generation or optimizer-quality claim'}
    dump_json(out/'manifest.json',manifest)
    tracker = None
    if args.wandb_mode != 'disabled':
        import wandb
        tracker=wandb.init(project=args.wandb_project, entity=args.wandb_entity,id=args.run_id,name='muon-adamw-fixed-weights-group-effects',
                          mode=args.wandb_mode,dir=str(out),config=manifest,tags=['muon','adamw','same-prefix'])
        tracker.log({'optimizer_step':0},step=0)
    exp=Experiment(model,tokenizer,out,tracker)
    cached={};results=[];states=[];warmup=[]
    for label,case in cases.items():
        restore(parameters,exp.initial)
        before,loss,norm=exp.residual(case)
        grads={n:p.grad.detach().clone() for n,p in parameters.items() if p.grad is not None}
        cached[label]={'before':before,'loss':loss,'norm':norm,'gradients':grads,
                      'group_norms':{g:vector_norm(grads[n] for n in names) for g,names in groups.items()}}
        exp.model.zero_grad(set_to_none=True)
    probe(exp,cases,cached,groups,None,0,results)
    restore(parameters,exp.initial)
    opts=optimizers(parameters,groups)
    warm_case={'rows':[{'origin':i,'family':r['family'],'input_ids':r['input_ids']} for i,r in enumerate(original['prefixes'])],
               'width':len(original['input_ids'][0]),'pad_id':cases['original-transition']['pad_id'],'target_origin':None}
    targets=torch.tensor(original['target_ids'],device='cuda',dtype=torch.long)
    for step in range(1,129):
        started=time.monotonic();model.zero_grad(set_to_none=True)
        z=exp.forward(warm_case,grad=True)
        with torch.autocast('cuda',enabled=False): loss=F.cross_entropy(z.float(),targets)
        loss.backward()
        group_norms={g:vector_norm(parameters[n].grad for n in names) for g,names in groups.items()}
        for opt in opts.values(): opt.step()
        torch.cuda.synchronize()
        elapsed=time.monotonic()-started
        exp.log_step('hybrid-supervised-history',loss=float(loss.detach()),warmup_optimizer_step=step,
                     muon_gradient_norm=group_norms['muon'],adamw_gradient_norm=group_norms['adamw'],
                     supervised_targets_per_second=len(targets)/elapsed)
        warmup.append({'step':step,'loss':float(loss.detach()),'group_gradient_l2':group_norms,'seconds':elapsed})
        if step in [16,128]:
            current=weights(parameters);snapshot=snapshot_optimizers(opts)
            statistics=state_statistics(opts)
            args.state_cache_dir.mkdir(parents=True,exist_ok=True)
            path=args.state_cache_dir/f'muon-checkpoint-{step}.pt'
            torch.save({'parameters':cpu_tree(current),'optimizers':cpu_tree(snapshot),
                        'groups':groups,'warmup_steps':step,'model':prepared['model'],'revision':prepared['revision'],
                        'configuration':manifest['muon'],'adamw_configuration':manifest['adamw']},path)
            os.chmod(path,0o600)
            states.append({'step':step,'path':str(path),'bytes':path.stat().st_size,
                           'sha256':digest_file(path),'statistics':statistics})
            dump_json(out/'hybrid-state-cache.json',states)
            probe(exp,cases,cached,groups,snapshot,step,results)
            restore(parameters,current)
            del current,snapshot
            dump_json(out/'warmup.json',warmup)
    summary={'status':'complete','hybrid_updates':exp.steps,'individual_optimizer_step_calls':2*exp.steps,
             'continuous_warmup_hybrid_updates':128,'probe_hybrid_updates':exp.steps-128,
             'paired_comparisons':len(results),'results':results,'group_info':group_info,
             'state_cache':states,'configuration':{'muon':manifest['muon'],'adamw':manifest['adamw']},
             'scope':manifest['scope']}
    dump_json(out/'summary.json',summary)
    if tracker is not None:
        tracker.summary.update({'status':'complete','hybrid_updates':exp.steps,'individual_optimizer_step_calls':2*exp.steps})
        tracker.finish()
    manifest.update({'status':'complete','hybrid_updates':exp.steps,
                     'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    dump_json(out/'manifest.json',manifest)
    if args.publish_dir:
        args.publish_dir.mkdir(parents=True,exist_ok=True)
        for name in ['summary.json','manifest.json','muon-comparisons.json','hybrid-state-cache.json']:
            shutil.copyfile(out/name,args.publish_dir/name)
        archive=out.parent/'muon-adamw-results.tar.gz'
        with tarfile.open(archive,'w:gz') as t: t.add(out,arcname='results')
        shutil.copyfile(archive,args.publish_dir/archive.name)
        dump_json(args.publish_dir/'export.json',{'archive_sha256':digest_file(archive),'archive_bytes':archive.stat().st_size,
                   'large_states_in_archive':False,'hybrid_updates':exp.steps})
    print(json.dumps({'status':'complete','hybrid_updates':exp.steps,'paired_comparisons':len(results)}),flush=True)


if __name__=='__main__': main()
