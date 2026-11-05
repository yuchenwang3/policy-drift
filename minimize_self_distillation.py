#!/usr/bin/env python3
"""Reduce the recorded Qwen self-distillation argmax flip and test Adam history.

No rollout or RL. Warmup consists of supervised updates on the eight existing
fixtures. Every numerical-residual probe has a zero-gradient paired control
starting from exactly the same parameters and cloned optimizer state.
"""
from __future__ import annotations

import argparse
import copy
import gc
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

from lab import dump_json, soft_loss, stable_log_softmax, write_csv
from real_model_step import digest_file, get_hidden, projected_logits, restore


@torch.no_grad()
def weights(parameters):
    return {name: p.detach().clone() for name, p in parameters.items()}


def vector_norm(tensors):
    return math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors))


@torch.no_grad()
def update_metrics(parameters, baseline):
    squared, changed = 0.0, 0
    for name, p in parameters.items():
        delta = p.detach().double() - baseline[name].double()
        squared += float(delta.square().sum())
        changed += int(torch.count_nonzero(delta))
    return {'update_l2': math.sqrt(squared), 'changed_parameter_coordinates': changed}


def kl(old, new):
    old, new = old.detach().double(), new.detach().double()
    p = stable_log_softmax(old).exp()
    delta = new - old
    d = delta - (p * delta).sum(-1, keepdim=True)
    series = d.square() * (0.5 + d * (1/6 + d * (1/24 + d * (1/120 + d/720))))
    remainder = torch.where(d.abs() < 0.01, series, torch.expm1(d) - d)
    values = torch.log1p((p * remainder).sum(-1))
    return {'kl_mean': float(values.mean()), 'kl_max': float(values.max()),
            'kl_per_prefix': values.cpu().tolist(),
            'different_logits': int(torch.count_nonzero(old != new)),
            'argmax_different_prefixes': int(torch.count_nonzero(old.argmax(-1) != new.argmax(-1)))}


def make_batch(case, device):
    sequences = [r['input_ids'] for r in case['rows']]
    width = case['width']
    if width < max(map(len, sequences)):
        raise ValueError('Width is shorter than a retained prefix')
    ids = torch.full((len(sequences), width), case['pad_id'], device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, tokens in enumerate(sequences):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        mask[i, :len(tokens)] = 1
    positions = torch.arange(width, device=device).expand_as(ids)
    last = mask.sum(-1) - 1
    return ids, mask, positions, last


def case_key(case):
    return json.dumps(case, sort_keys=True)


def size(case):
    return (len(case['rows']), sum(len(r['input_ids']) for r in case['rows']), case['width'])


class Experiment:
    def __init__(self, model, tokenizer, out, tracker, lr=1e-5, temperature=4.0):
        self.model, self.tokenizer, self.out, self.tracker = model, tokenizer, out, tracker
        self.parameters = dict(model.named_parameters())
        self.initial = weights(self.parameters)
        self.lr, self.temperature = lr, temperature
        self.steps = 0
        self.trials, self.cache, self.comparisons, self.warmup = [], {}, [], []
        self.original_pair = None
        self.best_any, self.best_pair = None, None

    def optimizer(self, state=None):
        opt = torch.optim.Adam(self.parameters.values(), lr=self.lr, eps=1e-8,
                               betas=(0.9, 0.999), weight_decay=0,
                               foreach=False, fused=False)
        if state is not None:
            # load_state_dict can reuse same-device tensors. Give each branch
            # a deep copy so its moment updates cannot change another branch.
            opt.load_state_dict(copy.deepcopy(state))
        return opt

    def log_step(self, phase, **metrics):
        self.steps += 1
        if self.tracker is not None:
            self.tracker.log({'optimizer_step': self.steps, 'phase': phase,
                              'learning_rate': self.lr, **metrics}, step=self.steps)

    def forward(self, case, grad=False):
        batch = make_batch(case, next(self.model.parameters()).device)
        with torch.set_grad_enabled(grad), torch.autocast('cuda', dtype=torch.bfloat16):
            h = get_hidden(self.model, *batch)
            return projected_logits(h, self.model.lm_head.weight)

    def residual(self, case, method='upcast'):
        self.model.zero_grad(set_to_none=True)
        z = self.forward(case, grad=True)
        teacher = (z.detach() / self.temperature)
        with torch.autocast('cuda', enabled=False):
            loss = soft_loss(z / self.temperature, teacher, method).mean()
        loss.backward()
        norm = vector_norm(p.grad for p in self.parameters.values() if p.grad is not None)
        return z.detach(), float(loss.detach()), norm

    def details(self, before, after):
        old_ids, new_ids = before.argmax(-1).tolist(), after.argmax(-1).tolist()
        top2 = torch.topk(before.double(), 2, dim=-1).values
        return {'before_argmax_ids': old_ids, 'after_argmax_ids': new_ids,
                'before_argmax_text': [self.tokenizer.decode([v]) for v in old_ids],
                'after_argmax_text': [self.tokenizer.decode([v]) for v in new_ids],
                'before_top2_logit_gaps': (top2[:, 0]-top2[:, 1]).cpu().tolist(),
                **kl(before, after)}

    def trial(self, case, label, initial=False):
        key = case_key(case)
        if key in self.cache:
            return self.cache[key]
        restore(self.parameters, self.initial)
        before, loss, norm = self.residual(case)
        opt = self.optimizer()
        opt.step()
        self.log_step('reduction', loss=loss, gradient_norm=norm)
        after = self.forward(case)
        target = next(i for i, r in enumerate(case['rows']) if r['origin'] == case['target_origin']) if case['target_origin'] is not None else None
        details = self.details(before, after)
        if initial:
            changed = [i for i, (a,b) in enumerate(zip(details['before_argmax_ids'], details['after_argmax_ids'])) if a != b]
            if len(changed) != 1:
                dump_json(self.out/'original-unexpected-transition.json', details)
                raise RuntimeError('Original replay did not yield exactly one changed prefix')
            target = changed[0]
            self.original_pair = (details['before_argmax_ids'][target], details['after_argmax_ids'][target])
            case['target_origin'] = case['rows'][target]['origin']
        pair = (details['before_argmax_ids'][target], details['after_argmax_ids'][target])
        row = {'trial': len(self.trials), 'label': label, 'case': copy.deepcopy(case),
               'batch_size': len(case['rows']), 'token_count': sum(len(r['input_ids']) for r in case['rows']),
               'width': case['width'], 'target_batch_index': target,
               'target_flipped': pair[0] != pair[1], 'same_original_transition': pair == self.original_pair,
               'loss': loss, 'gradient_l2': norm, **details,
               **update_metrics(self.parameters, self.initial)}
        self.trials.append(row)
        self.cache[case_key(case)] = row
        if initial:
            np.savez_compressed(self.out/'original-transition.npz',
                                before=before.float().cpu().numpy(), after=after.float().cpu().numpy())
        for kind, passed in [('any', row['target_flipped']), ('pair', row['same_original_transition'])]:
            previous = self.best_any if kind == 'any' else self.best_pair
            if passed and (previous is None or size(case) < size(previous['case'])):
                if kind == 'any': self.best_any = row
                else: self.best_pair = row
                np.savez_compressed(self.out/f'minimum-{kind}-transition.npz',
                                    before=before.float().cpu().numpy(), after=after.float().cpu().numpy())
                dump_json(self.out/f'minimum-{kind}-case.json', case)
        dump_json(self.out/'reduction-trials.json', self.trials)
        print(json.dumps({'phase': 'reduction', 'trial': row['trial'], 'label': label,
                          'size': size(case), 'target_flipped': row['target_flipped'],
                          'same_original_transition': row['same_original_transition'], 'pair': pair}), flush=True)
        del opt, before, after
        self.model.zero_grad(set_to_none=True)
        return row

    def reduce(self, original):
        first = self.trial(original, 'original-batch', initial=True)
        target_origin = original['target_origin']
        current = copy.deepcopy(original)
        only = copy.deepcopy(current)
        only['rows'] = [r for r in only['rows'] if r['origin'] == target_origin]
        if self.trial(only, 'only-original-trigger')['same_original_transition']:
            current = only
        else:
            # Delete batch rows while preserving the exact original argmax pair.
            changed = True
            while changed:
                changed = False
                for i, row in enumerate(current['rows']):
                    if row['origin'] == target_origin: continue
                    candidate = copy.deepcopy(current)
                    candidate['rows'].pop(i)
                    if self.trial(candidate, 'delete-batch-row')['same_original_transition']:
                        current, changed = candidate, True
                        break
        trimmed = copy.deepcopy(current)
        trimmed['width'] = max(len(r['input_ids']) for r in trimmed['rows'])
        if self.trial(trimmed, 'remove-right-padding')['same_original_transition']:
            current = trimmed
        # The original changed prefix has four tokens in this fixture. Enumerate
        # all its nonempty ordered subsequences instead of a heuristic search.
        target_index = next(i for i, r in enumerate(current['rows']) if r['origin'] == target_origin)
        tokens = current['rows'][target_index]['input_ids']
        if len(tokens) > 12:
            raise RuntimeError('Unexpected large trigger: explicit token-search scope required')
        for n in range(1, len(tokens)+1):
            for kept in itertools.combinations(range(len(tokens)), n):
                candidate = copy.deepcopy(current)
                candidate['rows'][target_index]['input_ids'] = [tokens[i] for i in kept]
                candidate['width'] = max(len(r['input_ids']) for r in candidate['rows'])
                self.trial(candidate, 'enumerate-trigger-subsequence')
        # Also shrink a one-row descendant without fixing the original argmax IDs.
        # It remains a self-distillation flip, but any changed token pair is labeled.
        single = copy.deepcopy(only)
        original_tokens = single['rows'][0]['input_ids']
        for n in range(1, len(original_tokens)+1):
            for kept in itertools.combinations(range(len(original_tokens)), n):
                candidate = copy.deepcopy(single)
                candidate['rows'][0]['input_ids'] = [original_tokens[i] for i in kept]
                candidate['width'] = n
                self.trial(candidate, 'single-prefix-subsequence')
        return first, self.best_pair, self.best_any

    def state_summary(self, opt):
        m2, v2, nzm, nzv, count, steps = 0.0, 0.0, 0, 0, 0, set()
        sample = {}
        for index, (parameter, state) in enumerate(opt.state.items()):
            m, v = state['exp_avg'], state['exp_avg_sq']
            m2 += float(m.double().square().sum())
            v2 += float(v.double().square().sum())
            nzm += int(torch.count_nonzero(m)); nzv += int(torch.count_nonzero(v))
            count += parameter.numel()
            steps.add(int(state['step']))
            if index in [0, 1, 50, 100]:
                sample[f'{index}:m'] = m.flatten()[:256].float().cpu().numpy()
                sample[f'{index}:v'] = v.flatten()[:256].float().cpu().numpy()
        fingerprint = hashlib.sha256(b''.join(sample[k].tobytes() for k in sorted(sample))).hexdigest()
        return {'steps': sorted(steps), 'first_moment_l2': math.sqrt(m2),
                'second_moment_l2': math.sqrt(v2), 'nonzero_first_moment_coordinates': nzm,
                'nonzero_second_moment_coordinates': nzv, 'state_parameter_coordinates': count,
                'sample_sha256': fingerprint}, sample

    def paired(self, case, baseline, state, label, warmup_step, state_kind):
        outcomes, zero_after = {}, None
        state_step = sorted({int(s['step']) for s in state['state'].values()}) if state else [0]
        before = None
        for gradient_kind in ['zero', 'native']:
            restore(self.parameters, baseline)
            self.model.zero_grad(set_to_none=True)
            opt = self.optimizer(state)
            if gradient_kind == 'native':
                current_before, loss, gradient_norm = self.residual(case)
            else:
                current_before, loss, gradient_norm = self.forward(case), None, 0.0
                for p in self.parameters.values():
                    p.grad = torch.zeros_like(p)
            if before is None: before = current_before
            elif not torch.equal(before, current_before):
                raise RuntimeError('Paired branches did not start from identical logits')
            opt.step()
            self.log_step('paired-probe', gradient_norm=gradient_norm,
                          probe_warmup_step=warmup_step, probe_label=label,
                          probe_gradient=gradient_kind)
            after = self.forward(case)
            outcomes[gradient_kind] = {'gradient_l2': gradient_norm, 'loss': loss,
                                       **update_metrics(self.parameters, baseline),
                                       **self.details(before, after)}
            if gradient_kind == 'zero':
                zero_after = weights(self.parameters)
                zero_logits = after.clone()
            else:
                difference = update_metrics(self.parameters, zero_after)
                pair_policy = self.details(zero_logits, after)
                np.savez_compressed(self.out/f'paired-{label}.npz',
                                    before=before.float().cpu().numpy(),
                                    zero_after=zero_logits.float().cpu().numpy(),
                                    native_after=after.float().cpu().numpy())
            del opt
            self.model.zero_grad(set_to_none=True)
        row = {'label': label, 'warmup_steps': warmup_step, 'state_kind': state_kind,
               'adam_step_before': state_step, 'zero_gradient_control': outcomes['zero'],
               'native_gradient_branch': outcomes['native'],
               'native_minus_zero_update_l2': difference['update_l2'],
               'native_minus_zero_changed_parameter_coordinates': difference['changed_parameter_coordinates'],
               'native_vs_zero_policy': pair_policy,
               'case': copy.deepcopy(case)}
        self.comparisons.append(row)
        dump_json(self.out/'moment-comparisons.json', self.comparisons)
        print(json.dumps({'phase': 'paired', 'label': label, 'adam_step_before': state_step,
                          'native_gradient_l2': outcomes['native']['gradient_l2'],
                          'extra_update_l2': difference['update_l2'],
                          'extra_policy_kl': pair_policy['kl_mean'],
                          'extra_argmax_changes': pair_policy['argmax_different_prefixes']}), flush=True)
        del zero_after, before, after, zero_logits
        restore(self.parameters, baseline)
        return row

    def stable_control(self, case, label):
        restore(self.parameters, self.initial)
        before, loss, gradient = self.residual(case, method='stable_upcast')
        opt = self.optimizer()
        opt.step()
        self.log_step('stable-control', loss=loss, gradient_norm=gradient)
        after = self.forward(case)
        row = {'loss': loss, 'gradient_l2': gradient,
               **update_metrics(self.parameters, self.initial), **self.details(before, after)}
        dump_json(self.out/f'minimum-stable-control-{label}.json', row)
        del opt, before, after
        self.model.zero_grad(set_to_none=True)
        return row

    def mature_states(self, cases, warmup_case, warmup_targets, checkpoints):
        restore(self.parameters, self.initial)
        stable = {}
        for case_label, case in cases.items():
            self.paired(case, self.initial, None, case_label+'-initial-fresh', 0, 'fresh')
            stable[case_label] = self.stable_control(case, case_label)
        restore(self.parameters, self.initial)
        opt = self.optimizer()
        for step in range(1, max(checkpoints)+1):
            self.model.zero_grad(set_to_none=True)
            z = self.forward(warmup_case, grad=True)
            with torch.autocast('cuda', enabled=False):
                loss = F.cross_entropy(z.float(), warmup_targets)
            loss.backward()
            gradient_norm = vector_norm(p.grad for p in self.parameters.values() if p.grad is not None)
            opt.step()
            self.log_step('supervised-warmup', loss=float(loss.detach()), gradient_norm=gradient_norm,
                          warmup_optimizer_step=step)
            self.warmup.append({'step': step, 'loss': float(loss.detach()), 'gradient_l2': gradient_norm,
                                'learning_rate': self.lr})
            if step in checkpoints:
                state_info, samples = self.state_summary(opt)
                np.savez_compressed(self.out/f'adam-state-samples-step-{step}.npz', **samples)
                dump_json(self.out/f'adam-state-step-{step}.json', state_info)
                current = weights(self.parameters)
                snapshot = copy.deepcopy(opt.state_dict())
                for case_label, case in cases.items():
                    # Actual warmup parameters, paired with accumulated moments.
                    self.paired(case, current, snapshot, f'{case_label}-current-warm-{step}', step, 'real-warmup')
                    # Same current weights, fresh moments: separates parameter drift.
                    self.paired(case, current, None, f'{case_label}-current-fresh-{step}', step, 'fresh-at-warmup-weights')
                    # Controlled transplant preserves the original counterexample's
                    # parameters. This is deliberately off the warmup trajectory.
                    self.paired(case, self.initial, snapshot, f'{case_label}-original-warm-{step}', step, 'moment-transplant')
                restore(self.parameters, current)
                del current, snapshot, samples
                write_csv(self.out/'warmup.csv', self.warmup)
                gc.collect()
        write_csv(self.out/'warmup.csv', self.warmup)
        return stable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--source-results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--warmup-checkpoints', type=int, nargs='+', default=[16, 128])
    parser.add_argument('--run-id', default=None, help='Explicit W&B run ID; omit for a fresh run.')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='offline')
    parser.add_argument('--wandb-project', default='numerical-policy-lab')
    parser.add_argument('--wandb-entity', default=None)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        parser.error('Existing experiment retained; choose a new output directory')
    prepared = json.loads(args.prepared.read_text())
    source = json.loads((args.source_results/'prefixes.json').read_text())
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(prepared['snapshot'], dtype=torch.bfloat16,
                                                attn_implementation='eager', local_files_only=True,
                                                trust_remote_code=False).float().cuda().eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(prepared['snapshot'], local_files_only=True,
                                             trust_remote_code=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    original = {'rows': [{'origin': i, 'family': r['family'], 'input_ids': r['input_ids']}
                         for i,r in enumerate(source['prefixes'])],
                'width': len(source['input_ids'][0]), 'pad_id': pad_id, 'target_origin': None}
    manifest = {'status': 'running', 'host': socket.gethostname(), 'pid': os.getpid(),
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'model': prepared['model'], 'revision': prepared['revision'],
                'source_results': str(args.source_results.resolve()), 'source_sha256': digest_file(__file__),
                'reused_lab_sha256': digest_file(Path(__file__).with_name('lab.py')),
                'reused_real_model_sha256': digest_file(Path(__file__).with_name('real_model_step.py')),
                'torch': torch.__version__, 'gpu_name': torch.cuda.get_device_name(),
                'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'learning_rate': 1e-5,
                'temperature': 4, 'temperature_squared_scaling': False,
                'adam': {'betas': [0.9, 0.999], 'epsilon': 1e-8, 'weight_decay': 0,
                         'foreach': False, 'fused': False},
                'master_parameters': 'FP32 loaded from pinned BF16 checkpoint',
                'autocast': 'BF16', 'warmup_checkpoints': args.warmup_checkpoints,
                'warmup_objective': 'hard-label next-token CE on the original eight fixtures',
                'probe_objective': 'same-model fixed-point soft-target CE, T=4',
                'scope': 'input reduction and controlled Adam-state experiment; no RL or generation'}
    dump_json(out/'manifest.json', manifest)
    tracker = None
    if args.wandb_mode != 'disabled':
        import wandb
        tracker = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=args.run_id,
                             name='self-distillation-minimum-and-adam-history', mode=args.wandb_mode,
                             dir=str(out), config=manifest, tags=['minimal-counterexample', 'adam-history'])
        tracker.log({'optimizer_step': 0}, step=0)
    exp = Experiment(model, tokenizer, out, tracker)
    # Reuse the captured source logits as a receipt for the exact original input.
    if not args.case:
        observed = exp.forward(original).float().cpu().numpy()
        expected = np.load(args.source_results/'captured-forward.npz')['logits']
        manifest['original_forward_bitwise_matches_saved'] = bool(np.array_equal(observed, expected))
        dump_json(out/'manifest.json', manifest)
        first, best_pair, best_any = exp.reduce(original)
        selected = copy.deepcopy(best_pair['case'])
        dump_json(out/'minimum-original-transition.json', best_pair)
        dump_json(out/'minimum-any-transition.json', best_any)
    else:
        selected = json.loads(args.case.read_text())
        exp.original_pair = (-1, -1)
        exp.trial(selected, 'supplied-minimum')
    for row in selected['rows']:
        row['decoded_text'] = tokenizer.decode(row['input_ids'])
    dump_json(out/'minimal-case.json', selected)
    targets = torch.tensor(source['target_ids'], dtype=torch.long, device='cuda')
    cases = {'original-transition': selected}
    if not args.case and case_key(best_any['case']) != case_key(best_pair['case']):
        cases['smallest-descendant'] = copy.deepcopy(best_any['case'])
    stable = exp.mature_states(cases, original, targets, sorted(set(args.warmup_checkpoints)))
    summary = {'status': 'complete', 'actual_optimizer_steps': exp.steps,
               'reduction_trial_count': len(exp.trials), 'minimal_case': selected,
               'original_argmax_pair': exp.original_pair,
               'minimum_same_original_transition': exp.best_pair,
               'minimum_any_transition': exp.best_any,
               'minimum_stable_control': stable, 'moment_comparisons': exp.comparisons,
               'probe_cases': cases,
               'warmup_actual_optimizer_steps': max(args.warmup_checkpoints),
               'minimality_scope': 'nonempty ordered subsequences of original changed prefix; '
                                  'batch deletions preserving that prefix; no weight/model-size search',
               'one_prefix_one_token_is_input_size_lower_bound': size(selected)[:2] == (1, 1)}
    dump_json(out/'summary.json', summary)
    if tracker is not None:
        tracker.summary.update({'status': 'complete', 'actual_optimizer_steps': exp.steps,
                                'warmup_steps': max(args.warmup_checkpoints)})
        tracker.finish()
    manifest.update({'status': 'complete', 'actual_optimizer_steps': exp.steps,
                      'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    dump_json(out/'manifest.json', manifest)
    print(json.dumps({'status': 'complete', 'actual_optimizer_steps': exp.steps,
                      'minimal_size': size(selected), 'host': socket.gethostname()}), flush=True)


if __name__ == '__main__':
    main()
