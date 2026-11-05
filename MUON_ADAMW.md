# Muon + AdamW parameter-group effects

In the tested grouping, the four-token input retains a residual-versus-zero argmax difference with 16- and 128-update histories. Group substitutions locate most of the accumulated-state policy effect in the Muon parameters. The two-token input has a distribution difference at those histories without an extra argmax difference.

The experiment restores original Qwen2.5-0.5B weights for every probe. Muon + AdamW builds its own state using 128 hard-label updates on the eight synthetic examples. Installing those states at original weights is a state-transplant diagnostic. It does not compare optimizer training quality.

Muon covers 168 hidden-layer matrices, 357,826,560 parameters. AdamW covers the remaining 122 tensors, 136,206,208 parameters, including tied embedding/head weights and all one-dimensional parameters. Tied weights appear once. Both base learning rates are 1e-5. PyTorch 2.11.0 Muon uses `match_rms_adamw`, momentum 0.95, Nesterov, epsilon 1e-7, and five Newton–Schulz iterations; its recorded orthogonalization path uses bf16. AdamW uses betas=(0.9,0.999) and epsilon=1e-8. Weight decay and clipping are disabled. The RMS preset does not imply equal realized update norms.

| Input / history | Muon extra update L2 | AdamW extra update L2 | Full extra KL | Argmax differences: full / Muon / AdamW |
|---|---:|---:|---:|---|
| 4-token / 0 | 2.814e-3 | 8.919e-4 | 2.3735 | 1 / 1 / 1 |
| 2-token / 0 | 2.152e-3 | 5.153e-4 | 0.2621 | 1 / 1 / 1 |
| 4-token / 16 | 6.379e-5 | 2.259e-6 | 0.1316 | 1 / 1 / 0 |
| 2-token / 16 | 4.148e-5 | 2.566e-6 | 0.01225 | 0 / 0 / 0 |
| 4-token / 128 | 3.643e-4 | 1.601e-6 | 0.6127 | 1 / 1 / 0 |
| 2-token / 128 | 1.764e-4 | 1.995e-6 | 0.001045 | 0 / 0 / 0 |

Each argmax denominator is one prefix. “Extra” always compares cloned-state residual and zero-gradient branches. At accumulated states, the Muon-only KL is nearly the full KL. AdamW-only KL is at most about 1.3e-13 and does not change argmax. Group KL values cannot be added.

To measure group policy effects, start from the zero branch's updated parameters and replace only one group's parameters with the residual branch's updated values, then run forward. Independent optimizer groups and the absence of cross-group clipping allow this substitution without additional optimizer steps. This measures an output effect rather than just parameter norms.

For the four-token input the zero-to-residual token pairs are `s → Auto` at state 16 and `Auto → m` at state 128. Both two-token branches predict `kit` at these histories. No better/worse label is assigned to either decision. The experiment does not isolate Muon momentum, normalization, or bf16 orthogonalization separately.

The run completed 140 mixed updates: 128 for continuous state accumulation and 12 paired probe updates. Each mixed update calls both optimizers, for 280 optimizer calls. [group-summary.csv](adam-ablation/muon-adamw/results/group-summary.csv) and the sanitized summary contain all six comparisons. Full branch logits and full mixed-state checkpoints are excluded from this compact release. The README gives the fresh CUDA command; no RL or long-run training benefit was measured.
