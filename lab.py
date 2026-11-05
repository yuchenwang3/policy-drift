#!/usr/bin/env python3
"""CPU numerical experiments. No pretrained models, downloads or GPU use.

References:
  https://jonbarron.info/softmaxfix/ (equations 5 and 9, adapted to PyTorch)

The logit-aware soft-target backward targets CE(softmax(z_T), softmax(z_S)).
It is NOT an implementation for an arbitrary, already-rounded target tensor.
The study explicitly records that target-representation distinction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time

import mpmath as mp
import numpy as np
import torch
import torch.nn.functional as F

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp32": torch.float32, "fp64": torch.float64}
METHODS = ("native", "upcast", "stable", "stable_upcast")
ROOT = Path(__file__).resolve().parent


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_saved_rows(path):
    """Recover completed sweep rows without recomputing them after an IO failure."""
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key in ('dtype', 'method', 'gradient', 'ideal_gradient'):
                continue
            if value in ('True', 'False'):
                row[key] = value == 'True'
            elif value == '':
                row[key] = None
            else:
                row[key] = json.loads(value)
    return rows


def promoted(z):
    return z if z.dtype == torch.float64 else z.float()


def stable_log_softmax(z):
    """Exclude the max slot before summing, retain its logprob with log1p.

    The max slot is a constant zero, rather than z_max - z_max. Autograd
    therefore obtains its derivative from the other slots. First-order only.
    """
    index = z.argmax(dim=-1, keepdim=True)
    peak = z.gather(-1, index)
    mask = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, index, True)
    shifted = torch.where(mask, torch.zeros_like(z), z - peak)
    others = torch.where(mask, torch.zeros_like(z), shifted.exp())
    return shifted - torch.log1p(others.sum(dim=-1, keepdim=True))


class StableSoftCE(torch.autograd.Function):
    """Stable first derivative of the logit-defined soft-target objective.

    The teacher is detached. Higher derivatives are outside this experiment.
    The large-displacement fallback follows the current project-page code.
    """
    @staticmethod
    def forward(ctx, student, teacher):
        logq = stable_log_softmax(student)
        q = logq.exp()
        p = stable_log_softmax(teacher).exp()
        delta = student - teacher
        centered = delta - (p * delta).sum(dim=-1, keepdim=True)
        log_ratio = centered - torch.log1p(
            (p * torch.expm1(centered)).sum(dim=-1, keepdim=True))
        small = log_ratio.abs() < 1
        # Avoid evaluating an overflowing expm1 in the unselected branch.
        local = p * torch.expm1(torch.where(small, log_ratio, torch.zeros_like(log_ratio)))
        difference = torch.where(small, local, q - p)
        grad = difference - q * difference.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(grad)
        return -(p * logq).sum(dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad_output.unsqueeze(-1) * grad, None


def hard_loss(z, method):
    if method == "native":
        return F.cross_entropy(z, torch.zeros(z.shape[0], dtype=torch.long), reduction="none")
    if method == "upcast":
        return F.cross_entropy(promoted(z), torch.zeros(z.shape[0], dtype=torch.long), reduction="none")
    if method == "stable_upcast":
        return -stable_log_softmax(promoted(z))[:, 0]
    return -stable_log_softmax(z)[:, 0]


def soft_loss(student, teacher, method):
    teacher = teacher.detach()
    if method == "stable":
        return StableSoftCE.apply(student, teacher)
    if method == "stable_upcast":
        return StableSoftCE.apply(promoted(student), promoted(teacher))
    if method == "upcast":
        student, teacher = promoted(student), promoted(teacher)
    return F.cross_entropy(student, F.softmax(teacher, dim=-1), reduction="none")


def mp_distribution(values):
    values = [mp.mpf(float(v)) for v in values]
    peak = max(values)
    exps = [mp.exp(v - peak) for v in values]
    total = mp.fsum(exps)
    return [v / total for v in exps], peak + mp.log(total)


def hard_reference(values):
    """80 decimal digits, on the exact values represented by the input dtype."""
    probs, lse = mp_distribution(values)
    target = -mp.fsum(probs[1:])  # Avoid even high-precision p_target - 1.
    grad = np.array([float(target), *[float(v) for v in probs[1:]]])
    margin = mp.mpf(float(values[0])) - mp.log(mp.fsum(mp.exp(mp.mpf(float(v))) for v in values[1:]))
    return grad, float(lse - mp.mpf(float(values[0]))), float(margin)


def soft_reference(student, teacher, stored_target):
    ps, _ = mp_distribution(student)
    pt, _ = mp_distribution(teacher)
    target = [mp.mpf(float(v)) for v in stored_target]
    mass = mp.fsum(target)
    ideal = np.array([float(a - b) for a, b in zip(ps, pt)])
    represented = np.array([float(mass * a - b) for a, b in zip(ps, target)])
    return ideal, represented, float(mass - 1)


def actual_step(initial, gradients, optimizer, lr=0.002, eps=1e-8):
    """A real torch.optim step, same fresh state and parameter precision."""
    param = torch.nn.Parameter(initial.detach().clone())
    param.grad = gradients.to(param.dtype).clone()
    if optimizer == "sgd":
        opt = torch.optim.SGD([param], lr=lr)
    else:
        opt = torch.optim.Adam([param], lr=lr, eps=eps, foreach=False, fused=False)
    opt.step()
    return (param.detach().double() - initial.double()).numpy()


def operator_experiment(out, dtype_names):
    mp.mp.dps = 80
    rows, cancellation = [], []
    requested = np.arange(0.0, 60.0001, 0.25)
    for name in dtype_names:
        dtype = DTYPES[name]
        master_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        for classes in (2, 16, 128):
            base = torch.zeros((len(requested), classes), dtype=torch.float64)
            base[:, 0] = torch.from_numpy(requested + math.log(classes - 1))
            quantized = base.to(dtype)
            reference = [hard_reference(z) for z in quantized.double().tolist()]
            ref_grad = torch.from_numpy(np.stack([r[0] for r in reference]))
            initial = quantized.to(master_dtype)
            ref_steps = {op: actual_step(initial, ref_grad, op) for op in ("sgd", "adam")}
            for method in METHODS:
                z = initial.clone().requires_grad_()
                losses = hard_loss(z.to(dtype), method)
                losses.sum().backward()  # Sum: each row is an independent example.
                gradient = z.grad.detach().double().numpy()
                updates = {op: actual_step(initial, z.grad, op) for op in ("sgd", "adam")}
                for i, margin in enumerate(requested):
                    rg, ref_loss, actual_margin = reference[i]
                    norm = np.linalg.norm(rg)
                    row = {"dtype": name, "classes": classes, "method": method,
                           "requested_margin": float(margin), "actual_margin": actual_margin,
                           "loss": float(losses[i]), "ref_loss": ref_loss,
                           "grad_target": float(gradient[i, 0]), "ref_grad_target": float(rg[0]),
                           "grad_target_relative_error": float(abs(gradient[i, 0] - rg[0]) / abs(rg[0])),
                           "grad_relative_l2_error": float(np.linalg.norm(gradient[i] - rg) / norm),
                           "grad_sum": float(gradient[i].sum()),
                           "grad_zero_target": bool(gradient[i, 0] == 0)}
                    for op in ("sgd", "adam"):
                        u, ru = updates[op][i], ref_steps[op][i]
                        row.update({f"{op}_update_l2": float(np.linalg.norm(u)),
                                    f"{op}_reference_update_l2": float(np.linalg.norm(ru)),
                                    f"{op}_update_error_l2": float(np.linalg.norm(u - ru)),
                                    f"{op}_different_coordinates": int(np.count_nonzero(u != ru)),
                                    f"{op}_target_update": float(u[0]),
                                    f"{op}_reference_target_update": float(ru[0])})
                    rows.append(row)
            print(f"hard sweep complete: {name}, K={classes}", flush=True)
        # Include equality and sub-ULP perturbations. Label input collapse explicitly.
        scales = np.r_[0.0, np.logspace(-20, -0.01, 161)]
        teacher = torch.tensor([2.0, 1.0, 0.0], dtype=dtype)
        students = (torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)[None, :]
                    + torch.from_numpy(scales)[:, None] * torch.tensor([-1.0, 1.0, 0.0])).to(dtype)
        for method in METHODS:
            initial = students.to(master_dtype)
            z = initial.clone().requires_grad_()
            teacher_rows = teacher.expand_as(students)
            losses = soft_loss(z.to(dtype), teacher_rows, method)
            losses.sum().backward()
            g = z.grad.double().numpy()
            target = F.softmax(promoted(teacher) if method in ("upcast", "stable_upcast") else teacher, -1).double().numpy()
            ideal_refs = []
            represented_refs = []
            masses = []
            for s in students.double().tolist():
                ideal, represented, mass = soft_reference(s, teacher.double().tolist(), target)
                ideal_refs.append(ideal)
                represented_refs.append(represented)
                masses.append(mass)
            ref_grad = torch.from_numpy(np.stack(ideal_refs))
            updates = {op: actual_step(initial, z.grad, op) for op in ("sgd", "adam")}
            ref_updates = {op: actual_step(initial, ref_grad, op) for op in ("sgd", "adam")}
            for i, eta in enumerate(scales):
                ideal, represented = ideal_refs[i], represented_refs[i]
                row = {"dtype": name, "method": method, "requested_perturbation": float(eta),
                       "actual_logit_difference_l2": float(np.linalg.norm(students[i].double().numpy() - teacher.double().numpy())),
                       "inputs_equal_after_quantization": bool(torch.equal(students[i], teacher)),
                       "gradient_l2": float(np.linalg.norm(g[i])),
                       "ideal_gradient_l2": float(np.linalg.norm(ideal)),
                       "gradient_error_l2": float(np.linalg.norm(g[i] - ideal)),
                       "stored_target_bias_l2": float(np.linalg.norm(represented - ideal)),
                       "stored_target_mass_error": masses[i],
                       "native_arithmetic_error_l2": float(np.linalg.norm(g[i] - represented)) if method in ("native", "upcast") else None,
                       "gradient_sum": float(g[i].sum()),
                       "gradient": json.dumps(g[i].tolist()), "ideal_gradient": json.dumps(ideal.tolist())}
                for op in ("sgd", "adam"):
                    u, ru = updates[op][i], ref_updates[op][i]
                    row.update({f"{op}_update_l2": float(np.linalg.norm(u)),
                                f"{op}_update_error_l2": float(np.linalg.norm(u - ru)),
                                f"{op}_different_coordinates": int(np.count_nonzero(u != ru))})
                cancellation.append(row)
    write_csv(out / "operator.csv", rows)
    write_csv(out / "cancellation.csv", cancellation)
    return rows, cancellation


def fixture(seed):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.integers(-8, 9, size=(32, 8)) / 8).double()
    magnitudes = rng.integers(1, 9, size=(9, 8)) / 8
    params = torch.from_numpy(magnitudes * rng.choice([-1.0, 1.0], size=(9, 8))).double()
    return torch.cat([x, torch.ones(32, 1, dtype=torch.float64)], dim=1), params


def exact_drift(x, params, initial, temperature):
    """FP64 evaluation on master parameters, prior to low-precision loss cast."""
    reference = x @ initial / temperature
    current = x @ params.detach().double() / temperature
    p = F.softmax(reference, dim=-1)
    delta = current - reference
    d = delta - (p * delta).sum(dim=-1, keepdim=True)
    # expm1(d)-d is O(d^2); use its series to retain tiny nonnegative KLs.
    series = d.square() * (0.5 + d * (1/6 + d * (1/24 + d * (1/120 + d/720))))
    remainder = torch.where(d.abs() < 0.01, series, torch.expm1(d) - d)
    kl = torch.log1p((p * remainder).sum(dim=-1)).mean()
    return float(kl), float(torch.linalg.vector_norm(d) / math.sqrt(d.numel())), float((current.argmax(-1) != reference.argmax(-1)).double().mean())


def self_distill(out, dtype_names, steps, seeds, lr, temperature, tracking):
    arms, first_steps, records = [], [], []
    for seed in seeds:
        x, initial = fixture(seed)
        np.savez(out / f"fixture-seed-{seed}.npz", x=x.numpy(), parameters=initial.numpy())
        for name in dtype_names:
            dtype = DTYPES[name]
            master_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            for optimizer in ("sgd", "adam"):
                for method in METHODS:
                    p = torch.nn.Parameter(initial.to(master_dtype).clone())
                    train_x = x.to(master_dtype)
                    teacher = ((train_x @ p.detach()) / temperature).to(dtype).detach()
                    if optimizer == "adam":
                        opt = torch.optim.Adam([p], lr=lr, eps=1e-8, foreach=False, fused=False)
                    else:
                        opt = torch.optim.SGD([p], lr=lr)
                    arm_id = f"{name}/{method}/{optimizer}/seed{seed}"
                    arms.append({"id": arm_id, "dtype": name, "method": method,
                                 "optimizer": optimizer, "seed": seed, "p": p, "x": train_x,
                                 "x64": x, "initial": p.detach().double().clone(), "teacher": teacher,
                                 "opt": opt, "last_grad_norm": 0.0, "last_update_norm": 0.0})
    started = time.monotonic()
    with (out / "distillation.csv").open("w", newline="") as f:
        fields = ["arm", "dtype", "method", "optimizer", "seed", "optimizer_step",
                  "loss", "parameter_drift_l2", "parameter_drift_relative_l2",
                  "canonical_kl", "centered_logit_drift_rms", "argmax_change_fraction",
                  "gradient_l2", "actual_update_l2", "changed_coordinates", "learning_rate"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(steps + 1):
            wb = {"optimizer_step": step}
            for arm in arms:
                p = arm["p"]
                logits = ((arm["x"] @ p) / temperature).to(DTYPES[arm["dtype"]])
                losses = soft_loss(logits, arm["teacher"], arm["method"])
                loss = losses.mean()
                diff = p.detach().double() - arm["initial"]
                drift = float(torch.linalg.vector_norm(diff))
                kl, logit_drift, flips = exact_drift(arm["x64"], p, arm["initial"], temperature)
                row = {"arm": arm["id"], "dtype": arm["dtype"], "method": arm["method"],
                       "optimizer": arm["optimizer"], "seed": arm["seed"], "optimizer_step": step,
                       "loss": float(loss.detach()), "parameter_drift_l2": drift,
                       "parameter_drift_relative_l2": drift / float(torch.linalg.vector_norm(arm["initial"])),
                       "canonical_kl": kl, "centered_logit_drift_rms": logit_drift,
                       "argmax_change_fraction": flips,
                       "gradient_l2": arm["last_grad_norm"], "actual_update_l2": arm["last_update_norm"],
                       "changed_coordinates": arm.get("last_changed", 0), "learning_rate": lr}
                writer.writerow(row)
                records.append(row)
                if step % 10 == 0 or step == steps:
                    for key in ("loss", "parameter_drift_l2", "canonical_kl", "gradient_l2", "actual_update_l2", "learning_rate"):
                        wb[f"{arm['id']}/{key}"] = row[key]
                if step == steps:
                    continue
                arm["opt"].zero_grad(set_to_none=True)
                loss.backward()
                arm["last_grad_norm"] = float(torch.linalg.vector_norm(p.grad.double()))
                before = p.detach().double().clone()
                arm["opt"].step()
                update = p.detach().double() - before
                arm["last_update_norm"] = float(torch.linalg.vector_norm(update))
                arm["last_changed"] = int(torch.count_nonzero(update))
                if step == 0:
                    grad_values = p.grad.double().numpy()
                    first_steps.append({"arm": arm["id"], "dtype": arm["dtype"],
                                        "method": arm["method"], "optimizer": arm["optimizer"],
                                        "seed": arm["seed"], "gradient_l2": arm["last_grad_norm"],
                                        "actual_update_l2": arm["last_update_norm"],
                                        "changed_coordinates": arm["last_changed"],
                                        "ideal_update_l2": 0.0, "fresh_optimizer": True,
                                        "master_dtype": str(p.dtype), "parameter_count": p.numel()})
                    np.savez(out / ("first-step-" + arm["id"].replace("/", "-") + ".npz"),
                             initial=before.numpy(), gradient=grad_values, update=update.numpy())
            if tracking and (step % 10 == 0 or step == steps):
                tracking.log(wb, step=step)
            if step % 100 == 0 or step == steps:
                f.flush()
                dump_json(out / "progress.json", {"phase": "distillation", "completed_optimizer_steps_per_arm": step,
                                                   "arms": len(arms), "elapsed_seconds": time.monotonic() - started})
                print(f"distillation: {step}/{steps}, {len(arms)} arms", flush=True)
    write_csv(out / "first_step.csv", first_steps)
    return records, first_steps


def summarize(out, hard, cancellation, drift, first):
    cliffs = []
    for name in sorted({r["dtype"] for r in hard}):
        for method in METHODS:
            subset = [r for r in hard if r["dtype"] == name and r["method"] == method and r["classes"] == 16]
            zeros = [r for r in subset if r["grad_zero_target"]]
            bad = [r for r in subset if r["grad_target_relative_error"] >= 0.5]
            cliffs.append({"dtype": name, "method": method,
                           "first_zero_margin_K16": zeros[0]["actual_margin"] if zeros else None,
                           "first_50pct_error_margin_K16": bad[0]["actual_margin"] if bad else None})
    equal = [r for r in cancellation if r["requested_perturbation"] == 0]
    last = max(r["optimizer_step"] for r in drift)
    final = [r for r in drift if r["optimizer_step"] == last]
    dump_json(out / "summary.json", {"hard_cliffs": cliffs, "equal_logits": equal,
                                    "first_steps": first, "final_distillation": final,
                                    "counts": {"hard_rows": len(hard), "cancellation_rows": len(cancellation),
                                               "distillation_rows": len(drift), "arms": len(first)}})


def main():
    global METHODS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--threads", type=int, default=1,
                        help="Fixed CPU operator condition; tiny matrices avoid multithread overhead.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb-project", default="policy-drift")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--reuse-operator-results", type=Path, default=None,
                        help="Reuse completed operator.csv and cancellation.csv after an unrelated failure.")
    args = parser.parse_args()
    METHODS = tuple(args.methods)
    if args.steps < 1 or args.temperature <= 0 or args.threads < 1:
        parser.error("steps, temperature and threads must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        parser.error("output directory already contains a run; use a fresh --out")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    torch.set_flush_denormal(False)
    config = {**vars(args), "out": str(out),
              "reuse_operator_results": str(args.reuse_operator_results) if args.reuse_operator_results else None,
              "device": "cpu", "master_parameter_policy": "fp32 except fp64 arm",
              "model": "linear classifier 9x8 (8 features plus bias), 32 fixed inputs",
              "weight_decay": 0.0, "dropout": 0.0, "batching": "full batch, fixed input order",
              "reference_decimal_digits": 80, "operator_reduction": "sum of independent rows",
              "distillation_reduction": "mean of 32 rows", "adam_epsilon": 1e-8,
              "tracking_scope": "one study run, per-arm metrics at original optimizer steps"}
    manifest = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
                "pid": os.getpid(), "torch": torch.__version__, "numpy": np.__version__,
                "mpmath": mp.__version__, "cpu_threads": torch.get_num_threads(),
                "cpu_count": os.cpu_count(), "config": config,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump_json(out / "manifest.json", manifest)
    tracking = None
    if args.wandb_mode != "disabled":
        import wandb
        tracking = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                              id=args.run_id, name="operator-and-self-distillation-cpu",
                              mode=args.wandb_mode, config=config, dir=str(out),
                              tags=["toy-numerical-experiment", "fixed-inputs", "cpu"])
        dump_json(out / "tracking.json", {"run_id": tracking.id, "mode_at_execution": args.wandb_mode,
                                          "directory": tracking.dir, "project": args.wandb_project,
                                          "entity": tracking.entity})
    if args.reuse_operator_results:
        hard = read_saved_rows(args.reuse_operator_results / "operator.csv")
        cancellation = read_saved_rows(args.reuse_operator_results / "cancellation.csv")
        write_csv(out / "operator.csv", hard)
        write_csv(out / "cancellation.csv", cancellation)
        print(f"Reused {len(hard)} hard-target rows and {len(cancellation)} cancellation rows", flush=True)
    else:
        hard, cancellation = operator_experiment(out, args.dtypes)
    drift, first = self_distill(out, args.dtypes, args.steps, args.seeds, args.lr, args.temperature, tracking)
    summarize(out, hard, cancellation, drift, first)
    if tracking:
        tracking.summary["completed_optimizer_steps_per_arm"] = args.steps
        tracking.summary["completed_arms"] = len(first)
        tracking.summary["status"] = "complete"
        tracking.finish()
    manifest.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "complete"})
    dump_json(out / "manifest.json", manifest)
    dump_json(out / "progress.json", {"phase": "complete", "completed_optimizer_steps_per_arm": args.steps,
                                       "arms": len(first)})
    print(json.dumps({"status": "complete", "host": socket.gethostname(), "out": str(out),
                      "arms": len(first), "optimizer_steps_per_arm": args.steps}), flush=True)


if __name__ == "__main__":
    main()
