#!/usr/bin/env python3
"""Same-prefix, single-update numerical study on a pinned pretrained model.

Reuses lab.py. No generation, rollout, RL objective, scheduler or multistep run.
The four controlled head VJPs share forward values and change one cast at a time.
They are diagnostic custom backwards, not claims about a stock backend.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab import dump_json, soft_loss, stable_log_softmax, write_csv

HARD_METHODS = ['pt_fp32', 'pt_native', 'stable_fp32', 'liger_native', 'liger_fp32',
                'pt_fp32_chunk1', 'liger_flce', 'ctrl_keep', 'ctrl_logits_round',
                'ctrl_hidden_round', 'ctrl_weight_round']
SOFT_METHODS = ['self_pt_native', 'self_pt_fp32', 'self_stable_fp32']
CONTROL_TRACES = {}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm(t):
    return float(torch.linalg.vector_norm(t.detach().double()))


def compare_tensor(actual, reference):
    a, b = actual.detach().double(), reference.detach().double()
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    error = torch.linalg.vector_norm(a - b)
    cosine = (torch.sum(a * b) / (an * bn)).item() if float(an) and float(bn) else None
    return {'norm': float(an), 'reference_norm': float(bn), 'error_l2': float(error),
            'relative_l2_error': float(error / bn) if float(bn) else None,
            'cosine': cosine, 'different_coordinates': int(torch.count_nonzero(a != b)),
            'lost_nonzero_coordinates': int(torch.count_nonzero((b != 0) & (a == 0))),
            'sign_flips': int(torch.count_nonzero(a * b < 0)), 'elements': a.numel()}


def compact_comparison(values):
    total = {k: 0.0 for k in ['a2', 'b2', 'error2', 'dot']}
    counts = {k: 0 for k in ['different_coordinates', 'lost_nonzero_coordinates', 'sign_flips', 'elements']}
    for a, b in values:
        x, y = a.detach().double(), b.detach().double()
        total['a2'] += float(torch.sum(x.square()))
        total['b2'] += float(torch.sum(y.square()))
        total['error2'] += float(torch.sum((x - y).square()))
        total['dot'] += float(torch.sum(x * y))
        counts['different_coordinates'] += int(torch.count_nonzero(x != y))
        counts['lost_nonzero_coordinates'] += int(torch.count_nonzero((y != 0) & (x == 0)))
        counts['sign_flips'] += int(torch.count_nonzero(x * y < 0))
        counts['elements'] += x.numel()
    return {'norm': math.sqrt(total['a2']), 'reference_norm': math.sqrt(total['b2']),
            'error_l2': math.sqrt(total['error2']),
            'relative_l2_error': math.sqrt(total['error2'] / total['b2']) if total['b2'] else None,
            'cosine': total['dot'] / math.sqrt(total['a2'] * total['b2']) if total['a2'] and total['b2'] else None,
            **counts}


class ControlledHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, cast_location, label):
        # Both operands are exactly those that the BF16 AMP projection consumes.
        hq, wq = hidden.to(torch.bfloat16), weight.to(torch.bfloat16)
        with torch.autocast('cuda', enabled=False):
            logits = F.linear(hq, wq)
        ctx.save_for_backward(hq, wq)
        ctx.hidden_dtype, ctx.weight_dtype = hidden.dtype, weight.dtype
        ctx.location, ctx.label = cast_location, label
        return logits.float()

    @staticmethod
    def backward(ctx, gradient):
        hq, wq = ctx.saved_tensors
        with torch.autocast('cuda', enabled=False):
            before = gradient.float()
            after = before.to(torch.bfloat16).float() if ctx.location == 'logits' else before
            dh_before = after @ wq.float()
            dw_before = after.t() @ hq.float()
            dh = dh_before.to(torch.bfloat16).float() if ctx.location == 'hidden' else dh_before
            dw = dw_before.to(torch.bfloat16).float() if ctx.location == 'weight' else dw_before
            CONTROL_TRACES[ctx.label] = {
                'only_changed_cast': ctx.location,
                'logits_gradient': compare_tensor(after, before),
                'hidden_gradient': compare_tensor(dh, dh_before),
                'head_weight_gradient': compare_tensor(dw, dw_before),
                'hidden_return_dtype': str(ctx.hidden_dtype), 'weight_return_dtype': str(ctx.weight_dtype),
                'backward_matmul': 'FP32, TF32 disabled; identical in all four controls'}
        return dh.to(ctx.hidden_dtype), dw.to(ctx.weight_dtype), None, None


def fixtures(tokenizer, device):
    specs = [
        ('text', 'The capital of France is', ' Paris.'),
        ('math', '2 + 2 =', ' 4.'),
        ('code', 'def add(a, b):\n    return a +', ' b\n'),
        ('tool_name', '{"tool": "', 'search", "arguments": {"query": "Paris"}}'),
        ('tool_argument', '{"tool": "search", "arguments": {"query": "', 'Paris"}}'),
        ('json_boundary', '{"ok": true, "count": 3', '}'),
        ('chinese', '法国的首都是', '巴黎。'),
        ('sequence', '1, 2, 3, 4, 5, 6, 7, 8, 9,', ' 10'),
    ]
    rows = []
    for family, text, continuation in specs:
        enc = tokenizer(text + continuation, add_special_tokens=False, return_offsets_mapping=True)
        index = next(i for i, (a, b) in enumerate(enc['offset_mapping']) if b > len(text))
        if index == 0:
            raise ValueError('Fixture has no nonempty token prefix')
        ids = enc['input_ids'][:index]
        target = enc['input_ids'][index]
        rows.append({'family': family, 'source_prefix_text': text, 'source_continuation': continuation,
                     'input_ids': ids, 'target_id': target,
                     'actual_prefix_text': tokenizer.decode(ids), 'target_text': tokenizer.decode([target]),
                     'target_offset': enc['offset_mapping'][index]})
    width = max(len(r['input_ids']) for r in rows)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row['input_ids'])] = torch.tensor(row['input_ids'], device=device)
        mask[i, :len(row['input_ids'])] = 1
    positions = torch.arange(width, device=device).expand_as(ids)
    targets = torch.tensor([r['target_id'] for r in rows], dtype=torch.long, device=device)
    last = mask.sum(-1) - 1
    return rows, ids, mask, positions, targets, last


def logit_reference(logits, targets):
    z = logits.double()
    logp = stable_log_softmax(z)
    probability = logp.exp()
    gradient = probability.clone()
    other = probability.clone().scatter(-1, targets[:, None], 0)
    gradient.scatter_(-1, targets[:, None], -other.sum(-1, keepdim=True))
    return -logp.gather(-1, targets[:, None]).mean(), gradient / len(targets)


def build_loss(logits, targets, method, ce):
    if method in ['pt_native', 'liger_native']:
        operand = logits.clone()
    else:
        operand = logits.float().clone()
    operand.retain_grad()
    if method in ['liger_native', 'liger_fp32']:
        loss = ce(operand, targets)
    elif method == 'stable_fp32':
        loss = -stable_log_softmax(operand).gather(-1, targets[:, None]).mean()
    else:
        loss = F.cross_entropy(operand, targets)
    return loss, operand


def isolated_ce(logits0, targets, ce, out):
    ref_loss, ref_grad = logit_reference(logits0, targets)
    rows = []
    for method in ['pt_native', 'pt_fp32', 'stable_fp32', 'liger_native', 'liger_fp32']:
        leaf = logits0.float().detach().clone().requires_grad_()
        boundary = leaf.to(torch.bfloat16)
        boundary.retain_grad()
        loss, operand = build_loss(boundary, targets, method, ce)
        loss.backward()
        before = operand.grad
        after = boundary.grad
        actual = compare_tensor(after, ref_grad)
        cast_reference = ref_grad.to(torch.bfloat16).double()
        np.savez_compressed(out / f'isolated-{method}.npz',
                            operand_gradient=before.detach().float().cpu().numpy(),
                            after_cast_gradient=after.detach().float().cpu().numpy())
        for i, target in enumerate(targets.tolist()):
            token_ref = ref_grad[i]
            token_actual = compare_tensor(after[i], token_ref)
            rows.append({'method': method, 'prefix_index': i, 'loss': float(loss),
                         'reference_loss': float(ref_loss), 'operand_dtype': str(operand.dtype),
                         'after_cast_dtype': str(after.dtype), 'target_gradient': float(after[i, target]),
                         'reference_target_gradient': float(token_ref[target]),
                         'relative_l2_error': token_actual['relative_l2_error'],
                         'lost_nonzero_coordinates': token_actual['lost_nonzero_coordinates'],
                         'whole_gradient_l2': actual['norm'],
                         'whole_cast_error_l2': norm(after.double() - before.double()) if before.dtype == torch.float32 else None,
                         'whole_error_beyond_expected_cast_l2': norm(after.double() - cast_reference),
                         'grad_fn': type(loss.grad_fn).__name__})
    write_csv(out / 'isolated-ce.csv', rows)
    return rows


def get_hidden(model, ids, mask, positions, last):
    full = model.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                       use_cache=False, return_dict=True).last_hidden_state
    return full[torch.arange(len(ids), device=ids.device), last].contiguous()


def projected_logits(hidden, weight, chunk1=False):
    if chunk1:
        return torch.cat([F.linear(hidden[i:i+1], weight) for i in range(len(hidden))], 0)
    return F.linear(hidden, weight)


@torch.no_grad()
def evaluate(model, ids, mask, positions, last, targets, old_logits):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = get_hidden(model, ids, mask, positions, last)
        new = projected_logits(h, model.lm_head.weight)
    old, new = old_logits.double(), new.double()
    old_logp, new_logp = stable_log_softmax(old), stable_log_softmax(new)
    p = old_logp.exp()
    delta = new - old
    d = delta - (p * delta).sum(-1, keepdim=True)
    series = d.square() * (0.5 + d * (1/6 + d * (1/24 + d * (1/120 + d/720))))
    remainder = torch.where(d.abs() < 0.01, series, torch.expm1(d) - d)
    kl = torch.log1p((p * remainder).sum(-1))
    target_change = (new_logp - old_logp).gather(-1, targets[:, None]).squeeze(-1)
    return {'policy_kl_mean': float(kl.mean()), 'policy_kl_max': float(kl.max()),
            'argmax_changed_prefixes': int(torch.count_nonzero(old.argmax(-1) != new.argmax(-1))),
            'target_logprob_abs_change_max': float(target_change.abs().max()),
            'target_logprob_changes': target_change.cpu().tolist(),
            'changed_logits': int(torch.count_nonzero(new != old))}


@torch.no_grad()
def restore(parameters, baseline):
    for name, parameter in parameters.items():
        parameter.copy_(baseline[name])


def gradient_summary(parameters, reference, method, output_rows):
    total = compact_comparison((p.grad, reference[name] if reference else torch.zeros_like(p.grad))
                               for name, p in parameters.items() if p.grad is not None)
    for name, parameter in parameters.items():
        if parameter.grad is None:
            continue
        ref = reference[name] if reference else torch.zeros_like(parameter.grad)
        comparison = compare_tensor(parameter.grad, ref)
        output_rows.append({'method': method, 'parameter': name, 'dtype': str(parameter.grad.dtype), **comparison})
    return total


def parameter_update(parameters, baseline, update_reference, store_reference=False):
    # Subtract in FP64 so changes between represented FP32 parameters are retained.
    current = {name: p.detach().double() - baseline[name].double() for name, p in parameters.items()}
    ref = (current if store_reference else update_reference) or {name: torch.zeros_like(v) for name, v in current.items()}
    comparison = compact_comparison((current[name], ref[name]) for name in current)
    comparison['changed_parameters'] = sum(int(torch.count_nonzero(v)) for v in current.values())
    return comparison, {k: v.clone() for k, v in current.items()} if store_reference else None


def save_parameter_samples(parameters, baseline, out, method, optimizer, writeback):
    selected = ['model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight',
                'model.layers.23.self_attn.q_proj.weight', 'model.norm.weight']
    arrays = {}
    for name in selected:
        if name not in parameters:
            continue
        p, old = parameters[name].detach().flatten()[:256], baseline[name].flatten()[:256]
        arrays[name + ':before'] = old.float().cpu().numpy()
        arrays[name + ':after'] = p.float().cpu().numpy()
    np.savez_compressed(out / f'samples-{method}-{optimizer}-{writeback}.npz', **arrays)


def model_steps(model, batch, targets, logits0, ce, flce, out, lr, temperature, tracker):
    ids, mask, positions, last = batch
    parameters = dict(model.named_parameters())
    baseline = {name: p.detach().clone() for name, p in parameters.items()}
    gradient_reference = None
    update_references = {}
    rows, gradient_rows, head_rows = [], [], []
    teacher = (logits0 / temperature).detach()
    step_count = 0
    for method in HARD_METHODS + SOFT_METHODS:
        started = time.monotonic()
        restore(parameters, baseline)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = get_hidden(model, ids, mask, positions, last)
            h.retain_grad()
            chunk1 = method == 'pt_fp32_chunk1'
            z = projected_logits(h, model.lm_head.weight, chunk1=chunk1)
            z_before = z.detach().clone()
            same_logits = torch.equal(z_before, logits0)
            operand = None
            if method.startswith('ctrl_'):
                location = {'ctrl_keep': 'none', 'ctrl_logits_round': 'logits',
                            'ctrl_hidden_round': 'hidden', 'ctrl_weight_round': 'weight'}[method]
                diagnostic = ControlledHead.apply(h, model.lm_head.weight, location, method)
                if not torch.equal(diagnostic.detach(), logits0.float()):
                    raise RuntimeError('Controlled head forward differs; cast attribution invalid')
                operand = diagnostic
                operand.retain_grad()
                with torch.autocast('cuda', enabled=False):
                    loss = F.cross_entropy(operand, targets)
            elif method == 'liger_flce':
                loss = flce(model.lm_head.weight, h, targets)
            elif method in SOFT_METHODS:
                if not same_logits:
                    raise RuntimeError('Self-distillation initial logits differ from saved teacher')
                mapping = {'self_pt_native': 'native', 'self_pt_fp32': 'upcast',
                           'self_stable_fp32': 'stable_upcast'}
                z.retain_grad()
                with torch.autocast('cuda', enabled=False):
                    loss = soft_loss(z / temperature, teacher, mapping[method]).mean()
            else:
                z.retain_grad()
                with torch.autocast('cuda', enabled=False):
                    loss, operand = build_loss(z, targets,
                                               'pt_fp32' if method == 'pt_fp32_chunk1' else method, ce)
        if not method in ['pt_fp32_chunk1', 'liger_flce'] and not same_logits:
            raise RuntimeError(f'{method}: initial forward logits changed')
        loss.backward()
        if method == 'pt_fp32':
            gradient_reference = {name: p.grad.detach().clone() for name, p in parameters.items() if p.grad is not None}
        gradient = gradient_summary(parameters, gradient_reference if method in HARD_METHODS else None,
                                    method, gradient_rows)
        head = {'method': method, 'objective': 'hard_next_token' if method in HARD_METHODS else 'self_distillation',
                'loss': float(loss.detach()), 'same_observed_logits_as_baseline': same_logits,
                'hidden_dtype': str(h.dtype), 'head_weight_dtype': str(model.lm_head.weight.dtype),
                'logits_dtype': str(z.dtype), 'loss_dtype': str(loss.dtype),
                'loss_grad_fn': type(loss.grad_fn).__name__, 'hidden_gradient_norm': norm(h.grad),
                'parameter_gradient_norm': gradient['norm'], 'parameter_gradient_reference':
                    'pt_fp32' if method in HARD_METHODS else 'zero ideal fixed-point gradient',
                'parameter_gradient_relative_error': gradient['relative_l2_error'],
                'parameter_gradient_cosine': gradient['cosine'],
                'parameter_gradient_sign_flips': gradient['sign_flips']}
        if operand is not None and operand.grad is not None:
            head['loss_operand_gradient_dtype'] = str(operand.grad.dtype)
            head['loss_operand_gradient_norm'] = norm(operand.grad)
        head_rows.append(head)
        write_csv(out / 'head-boundaries.csv', head_rows)
        write_csv(out / 'parameter-gradients.csv', gradient_rows)
        for optimizer in ['sgd', 'adam']:
            restore(parameters, baseline)
            opt = (torch.optim.SGD(parameters.values(), lr=lr) if optimizer == 'sgd'
                   else torch.optim.Adam(parameters.values(), lr=lr, eps=1e-8, foreach=False, fused=False))
            opt.step()
            step_count += 1
            for writeback in ['fp32_master', 'bf16_writeback']:
                if writeback == 'bf16_writeback':
                    with torch.no_grad():
                        for p in parameters.values():
                            p.copy_(p.to(torch.bfloat16).float())
                reference_key = (optimizer, writeback)
                reference = update_references.get(reference_key) if method in HARD_METHODS else None
                update, saved = parameter_update(parameters, baseline, reference,
                                                  store_reference=(method == 'pt_fp32'))
                if saved is not None:
                    update_references[reference_key] = saved
                policy = evaluate(model, ids, mask, positions, last, targets, logits0)
                row = {'method': method, 'objective': head['objective'], 'optimizer': optimizer,
                       'writeback': writeback, 'optimizer_steps': 1, 'learning_rate': lr,
                       'loss_before': head['loss'], 'gradient_norm': gradient['norm'],
                       'gradient_relative_error': gradient['relative_l2_error'],
                       'update_norm': update['norm'], 'changed_parameters': update['changed_parameters'],
                       'update_error_vs_reference_l2': update['error_l2'],
                       'update_relative_error_vs_reference': update['relative_l2_error'],
                       'update_different_coordinates': update['different_coordinates'],
                       'update_reference': 'pt_fp32 same optimizer/writeback' if method in HARD_METHODS else 'ideal zero update',
                       'policy_kl_mean': policy['policy_kl_mean'], 'policy_kl_max': policy['policy_kl_max'],
                       'argmax_changed_prefixes': policy['argmax_changed_prefixes'],
                       'target_logprob_abs_change_max': policy['target_logprob_abs_change_max'],
                       'changed_logits': policy['changed_logits'],
                       'peak_owned_memory_GiB': torch.cuda.max_memory_allocated() / 2**30,
                       'elapsed_seconds': time.monotonic() - started}
                rows.append(row)
                save_parameter_samples(parameters, baseline, out, method, optimizer, writeback)
                dump_json(out / f'policy-{method}-{optimizer}-{writeback}.json', policy)
            del opt
            gc.collect()
        write_csv(out / 'one-step.csv', rows)
        dump_json(out / 'control-cast-traces.json', CONTROL_TRACES)
        dump_json(out / 'progress.json', {'completed_methods': len(head_rows), 'methods': len(HARD_METHODS + SOFT_METHODS),
                                         'actual_optimizer_steps': step_count, 'last_method': method})
        print(json.dumps({'method': method, 'gradient_norm': gradient['norm'], 'rows': len(rows),
                          'seconds': time.monotonic() - started}), flush=True)
        del loss, h, z, z_before, operand
    if tracker:
        metrics = {'optimizer_step': 1, 'completed_methods': len(head_rows), 'actual_optimizer_steps': step_count}
        for row in rows:
            prefix = '/'.join([row['method'], row['optimizer'], row['writeback']])
            for key in ['loss_before', 'gradient_norm', 'update_norm', 'policy_kl_mean', 'changed_parameters', 'learning_rate']:
                metrics[prefix + '/' + key] = row[key]
        tracker.log(metrics, step=1)
    return rows, head_rows, step_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--run-id', default=None, help='Explicit W&B run ID; omit for a fresh run.')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='offline')
    parser.add_argument('--wandb-project', default='policy-drift')
    parser.add_argument('--wandb-entity', default=None)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text())
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        parser.error('Refusing to overwrite an existing run')
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(prepared['snapshot'], local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(prepared['snapshot'], dtype=torch.bfloat16,
                                                attn_implementation='eager', local_files_only=True,
                                                trust_remote_code=False).float().to(device).eval()
    model.config.use_cache = False
    f, ids, mask, positions, targets, last = fixtures(tokenizer, device)
    dump_json(out / 'prefixes.json', {'prefixes': f, 'input_ids': ids.cpu().tolist(),
                                     'attention_mask': mask.cpu().tolist(), 'position_ids': positions.cpu().tolist(),
                                     'target_ids': targets.cpu().tolist(), 'last_prefix_positions': last.cpu().tolist()})
    manifest = {'status': 'running', 'host': socket.gethostname(), 'pid': os.getpid(),
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'model': prepared['model'], 'revision': prepared['revision'],
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'tied_embeddings': model.lm_head.weight is model.model.embed_tokens.weight,
                'device_name': torch.cuda.get_device_name(), 'compute_capability': torch.cuda.get_device_capability(),
                'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
                'liger': prepared['liger_version'], 'triton': importlib.metadata.version('triton'),
                'prefix_count': len(f), 'learning_rate': args.lr, 'adam_epsilon': 1e-8,
                'master_dtype': 'float32, loaded from a BF16 checkpoint', 'autocast_dtype': 'bfloat16',
                'attention': 'eager', 'use_cache': False, 'dropout': 'model.eval()', 'weight_decay': 0,
                'loss_autocast': 'disabled for logits-only CE; explicit operand dtype; enabled inside fused projection',
                'source_sha256': digest_file(__file__), 'reused_lab_sha256': digest_file(Path(__file__).with_name('lab.py')),
                'kernel_sources': prepared['sources'],
                'weight_files': {p.name: digest_file(p) for p in Path(prepared['snapshot']).glob('*.safetensors')},
                'hard_methods': HARD_METHODS, 'soft_methods': SOFT_METHODS,
                'scope': 'same-prefix numerical one-step study, no RL or generation'}
    dump_json(out / 'manifest.json', manifest)
    tracker = None
    if args.wandb_mode != 'disabled':
        import wandb
        tracker = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=args.run_id,
                              name='qwen05b-same-prefix-ce-cast-one-step', mode=args.wandb_mode, dir=str(out),
                              config=manifest, tags=['real-model', 'one-step', 'fused-ce', 'cast-boundary'])
        tracker.log({'optimizer_step': 0}, step=0)
    ce = LigerCrossEntropyLoss(reduction='mean')
    flce = LigerFusedLinearCrossEntropyLoss(reduction='mean', accum_dtype=torch.float32)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        h0 = get_hidden(model, ids, mask, positions, last)
        logits0 = projected_logits(h0, model.lm_head.weight).detach()
        chunk1 = projected_logits(h0, model.lm_head.weight, chunk1=True).detach()
    row_info = []
    probability = stable_log_softmax(logits0.double()).exp()
    for i, row in enumerate(f):
        target = int(targets[i])
        top2 = torch.topk(logits0[i].double(), 2).values
        row_info.append({'prefix_index': i, 'family': row['family'], 'target_text': row['target_text'],
                         'target_probability': float(probability[i, target]),
                         'argmax_probability': float(probability[i].max()),
                         'top2_logit_gap': float(top2[0] - top2[1]),
                         'chunk1_max_logit_difference': float((chunk1[i] - logits0[i]).abs().max())})
    write_csv(out / 'initial-probabilities.csv', row_info)
    np.savez_compressed(out / 'captured-forward.npz', logits=logits0.float().cpu().numpy(),
                        hidden=h0.float().cpu().numpy(), chunk1_logits=chunk1.float().cpu().numpy(),
                        targets=targets.cpu().numpy())
    isolated = isolated_ce(logits0, targets, ce, out)
    rows, heads, steps = model_steps(model, (ids, mask, positions, last), targets, logits0,
                                     ce, flce, out, args.lr, args.temperature, tracker)
    summary = {'model': prepared['model'], 'revision': prepared['revision'],
               'isolated_ce_rows': len(isolated), 'gradient_methods': len(heads),
               'actual_optimizer_steps': steps, 'update_evaluation_rows': len(rows),
               'self_stable_exact_zero_updates': all(r['update_norm'] == 0 for r in rows if r['method'] == 'self_stable_fp32'),
               'head': heads, 'updates': rows, 'initial_probabilities': row_info, 'cast_controls': CONTROL_TRACES}
    dump_json(out / 'summary.json', summary)
    if tracker is not None:
        tracker.summary.update({'optimizer_steps_per_arm': 1, 'actual_optimizer_steps': steps,
                                'completed_methods': len(heads), 'status': 'complete'})
        tracker.finish()
    manifest.update({'status': 'complete', 'actual_optimizer_steps': steps,
                      'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    dump_json(out / 'manifest.json', manifest)
    print(json.dumps({'status': 'complete', 'actual_optimizer_steps': steps,
                      'methods': len(heads), 'rows': len(rows), 'host': socket.gethostname()}), flush=True)


if __name__ == '__main__':
    main()
