# Adam component ablations

Historical second moments are the main suppressor of residual-update amplitude in these examples. First moments and step also move the zero-gradient branch relative to a decision boundary, so a small extra update can still change argmax.

The experiment holds the original public Qwen2.5-0.5B weights fixed, reuses the two minimal inputs, and computes each native fp32 residual once. It independently retains or clears `m`, `v`, and `step` from 16/128-update histories. Every cell compares a residual branch with an explicit zero-gradient branch from the same state. These are counterfactual diagnostic states, not ordinary training histories.

| Retained components | 4-token / 16 | 4-token / 128 | 2-token / 16 | 2-token / 128 |
|---|---:|---:|---:|---:|
| None | 1 | 1 | 1 | 1 |
| Step only | 1 | 1 | 1 | 1 |
| v only | 0 | 0 | 0 | 0 |
| v + step | 0 | 0 | 0 | 0 |
| m only | 1 | 0 | 1 | 0 |
| m + step | 1 | 0 | 1 | 0 |
| m + v | 1 | 0 | 0 | 0 |
| All | 0 | 0 | 0 | 0 |

Values are residual-versus-zero argmax differences, each over one prefix. The 32 rows include four reused fresh-state baselines. The run made 184 new optimizer updates: 128 for state reconstruction and 56 for paired probes. These updates do not form a single 184-step training trajectory.

## Amplitude and boundaries

Retaining only `v` reduces extra update L2 from 3.435e-2 to 4.446e-6 / 4.578e-6 for the four-token input, and from 1.180e-2 to 7.331e-6 / 7.529e-6 for the two-token input. Values are for histories 16 / 128. All four extra policy KL values are zero. At every nonzero residual coordinate, the historical contribution β₂v exceeds the new contribution (1−β₂)g².

Keeping only step reduces the corresponding updates to 7.706e-3 / 8.840e-3 and 2.062e-3 / 2.084e-3, while all four argmax differences remain. With initial moments zero and prior count t, the next scalar update is

```text
a_t = (1 - beta1) / (1 - beta1**(t + 1))
b_t = sqrt((1 - beta2) / (1 - beta2**(t + 1)))
delta = -lr * a_t * g / (b_t * abs(g) + epsilon)
```

The effective epsilon after factoring out b_t is 1e-8, 4.107e-8, and 1.100e-7 at t=0,16,128. This explains an amplitude change, not a token prediction.

Keeping 16-step `m,v` but resetting step puts the four-token zero branch at an exact maximum tie between `http` and `tool`. The residual branch selects `tool` with a top-two logit gap of 0.0625. Extra update L2 is only 6.164e-6 and extra KL is 5.544e-4. Conversely, retaining `m` while clearing `v` can create huge updates; some 128-step cases converge to the same `Paris` argmax despite parameter differences of tens in L2. An unchanged argmax alone does not demonstrate suppression.

The settings remain Adam lr=1e-5, epsilon=1e-8, betas=(0.9,0.999), no weight decay or clipping, fp32 master parameters, bf16 forward, and distillation temperature 4 without T². Policy KL is evaluated from actual forward logits in fp64.

[ablation-summary.csv](adam-ablation/results/ablation-summary.csv) contains all branch comparisons and top-two margins. The selected summary, denominator diagnostics, and branch-logit NPZ files support `analyze_adam_ablation.py`; see the README command. Large full-state checkpoints are excluded. Reconstructing their history matched prior aggregate norms, nonzero counts, and fixed samples; full historical bitwise state equivalence was not available.
