# CPU operator and synthetic-model baseline

The CPU baseline separates a loss-gradient error, an actual parameter writeback, a probability change, and an argmax change. A numerical residual can affect one without affecting all four.

The recorded environment used Python 3.12 and PyTorch 2.11.0+cu130, with every computation explicitly on CPU. High-precision operator references use mpmath at 80 decimal digits. The self-distillation model is synthetic and requires no pretrained weights.

## Two direct examples

For a 16-class hard-label example at requested margin 18 with fp32 input, native CE returns target gradient zero while the reference is −1.522998601493164e-8. Real SGD writeback changes that target coordinate by zero for both native and stable paths. Fresh Adam changes it by zero for native CE and +0.0012073516845703125 for stable/reference gradients. Learning rate is 0.002 and Adam epsilon is 1e-8.

For fp32 synthetic self-distillation, the native initial parameter-gradient L2 median is about 3.18e-9 across three seeds. Fresh SGD leaves parameters unchanged on the first step. Fresh Adam changes all 72 parameter coordinates, with median update L2=6.0144552e-4. Stable loss leaves initial gradient, first update, and step-1000 drift at zero.

| Loss-boundary precision / path | Median Adam first update L2 | Median first quantized-policy KL | Median step-1000 parameter drift L2 |
|---|---:|---:|---:|
| bf16 / native | 1.693e-2 | 4.712e-7 | 5.770 |
| bf16 / native upcast to fp32 | 6.014e-4 | 9.422e-11 | 9.007e-3 |
| fp16 / native | 1.678e-2 | 2.298e-7 | 5.951 |
| fp16 / native upcast to fp32 | 0 | 0 | 0 |
| fp32 / native | 6.014e-4 | 3.224e-10 | 6.366e-3 |
| fp64 / native | 1.040e-12 | 9.793e-28 | 1.079e-3 |
| stable or stable_upcast / all four precisions | 0 | 0 | 0 |

All 96 arms preserve argmax on the 32 fixed inputs after the first step. For bf16/native, at least 99.991% of the final parameter-drift squared norm lies in a common-class logit-shift direction, from a conservative singular-value bound. Its large parameter drift therefore needs to be read alongside the measured policy differences. [final_gauge_drift_bounds.csv](results/final_gauge_drift_bounds.csv) gives per-seed bounds.

## Precision boundaries

The hard-label scan covers 2, 16, and 128 classes, four input dtypes, and four methods. All methods receive the same represented input values; references use those values too. Target reference gradients are computed as the negative sum of non-target probabilities to avoid `p_target - 1` cancellation.

For the 16-class scan, the first observed zero target gradient occurs at these actual aggregate margins. Requested margins run from 0 to 60 in increments of 0.25; input quantization changes actual values. These are grid observations for this CPU implementation, not universal thresholds.

| Input dtype | Native | Native in fp32 | Stable in input dtype | Stable in fp32 |
|---|---:|---:|---:|---:|
| bf16 | 5.73 | 16.04 | None in scan | None in scan |
| fp16 | 7.75 | 16.25 | 14.75 | 17.50 |
| fp32 | 16.25 | 16.25 | None in scan | None in scan |
| fp64 | 34.75 | 34.75 | None in scan | None in scan |

Stable formulas cannot recover information lost at an input or gradient-cast boundary. In fp16, individual non-target gradients can underflow before their sum constructs the target gradient.

The near-fixed-point scan uses teacher logits `[2,1,0]` and student perturbation `eta * [-1,1,0]`, including exact equality and sub-ULP perturbations. `cancellation.csv` records whether represented student logits equal the teacher. For rounded stored target p_hat with mass s, the exact stored-target CE gradient is `s * softmax(student) - p_hat`. The table distinguishes this target representation bias from native backward arithmetic error. A stable row's target-bias diagnostic refers to a corresponding native target constructor, not a tensor consumed by stable backward.

## Synthetic training setup

Each seed fixes 32 inputs, eight features, eight classes, and a 9 × 8 parameter matrix including bias. The teacher and initial student are identical. Inputs and initial parameters are exactly representable binary rationals, saved in `fixture-seed-*.npz`. There is no dropout, decay, scheduler, or sampling. Distillation temperature is 4.

This isolates loss-boundary precision: bf16/fp16/fp32 arms have fp32 master parameters and forward, followed by the logit cast at loss entry; fp64 arms use fp64 throughout. It is not whole-network low-precision training. SGD and Adam use fresh state and lr=0.002; Adam epsilon is 1e-8.

`native` uses PyTorch CE. `upcast` forms targets and computes CE in fp32. `stable` applies the attributed logit-aware stable formula in input dtype. `stable_upcast` uses the stable formula in fp32. All fp64 arms retain fp64. The stable target is defined by teacher logits, requires those logits, and does not promise higher-order derivatives.

The 96 arms comprise 4 dtypes × 4 methods × 2 optimizers × 3 seeds, with 1,000 actual optimizer steps each. Saved CPU results contain 11,568 operator rows, 2,592 cancellation rows, and 96,096 training rows including initialization. Long-curve `canonical_kl` evaluates master parameters in fp64 before casting. First-step quantized policy KL evaluates the executed dtype-cast logits. These are different metrics. A training row's step is the count of completed updates; its gradient/update fields describe the update just completed. Initial gradients are in `first_step.csv`.

The [README](README.md) gives the exact CPU reproduction command and saved-result analysis commands. Changing PyTorch build or CPU can change last bits and observed cliff locations. The experiments establish a controlled gradient-to-update-to-policy chain; they do not measure RL reward or downstream quality.
