# Mean Prediction and Representation Dynamics in Toy JEPA

This repository investigates **mean prediction**, **representation variance**, and **linear recoverability** in a simplified Joint-Embedding Predictive Architecture (JEPA). It does **not** implement full I-JEPA, and the experiments do not establish pathological collapse in JEPA generally.

## Research questions

1. When does an MSE predictor output a nearly constant (mean) prediction?
2. Does the same behavior occur in a toy JEPA with encoders?
3. How does target-encoder representation variance change during training?
4. Does a decrease in representation variance coincide with a loss of linearly recoverable input information?
5. Does mean prediction persist when the context contains exploitable nonlinear predictive information?

## Experimental overview

| Experiment | Main test | Key observation |
| :--- | :--- | :--- |
| Exp01 | Conditional-mean prediction with a direct MLP | Mean prediction is appropriate when context is uninformative. |
| Exp02 | Frozen vs. EMA target encoders in toy JEPA | Constant-like predictions at `rho = 0` do not require constant target embeddings. |
| Exp03 | Target representation variance during training | EMA variance can shrink or grow depending on data correlation. |
| Exp04 | Linear recovery of target inputs | Reduced variance did not reduce linear recoverability in this setup. |
| Exp05 | Predictable nonlinear target | Toy JEPA learned the available predictive relationship across three seeds. |

## Common setup

Generate independent standard-normal variables `X` and `epsilon`, then define:

$$
Y = \rho X + \sqrt{1-\rho^2}\,\epsilon,
\qquad \rho \in \{0, 0.5, 1\}.
$$

- `rho = 0`: `X` and `Y` are independent.
- `rho = 0.5`: `X` provides partial information about `Y`.
- `rho = 1`: `Y = X`.

For Exp02–05, the toy JEPA consists of a context encoder, target encoder, and predictor:

$$
z_x = f_\theta(X), \qquad z_y = f_{\bar\theta}(Y), \qquad \hat z_y = g_\phi(z_x).
$$

The predictor and context encoder minimize:

$$
\mathcal L = \mathbb E\left[\|\hat z_y-z_y\|_2^2\right].
$$

The target encoder is either **frozen** or updated by **exponential moving average (EMA)**:

$$
\bar\theta \leftarrow \tau\bar\theta+(1-\tau)\theta, \qquad \tau=0.99.
$$

> **Terminology:** A nearly constant predictor output is not the same as encoder representation collapse. When `X` contains no information about `Y`, mean prediction can be the optimal MSE solution rather than a training failure.

## Exp01 — Mean prediction under MSE

**Script:** `experiments/exp01_mean_prediction.py`

**Why:** Establish the behavior of a plain MLP predictor before adding encoders.

For squared-error regression, the optimal prediction is the conditional mean:

$$
g^*(X)=\mathbb E[Y\mid X]=\rho X.
$$

Thus, at `rho = 0`, the optimal output is the constant mean `0`.

| rho | Prediction MSE | Prediction variance | Mean-baseline MSE |
| ---: | ---: | ---: | ---: |
| 0.0 | 1.0269 | 0.0044 | 1.0223 |
| 0.5 | 0.7707 | 0.2462 | 1.0139 |
| 1.0 | 0.0000 | 0.9962 | 0.9962 |

**What we learned:** The predictor produces nearly constant outputs when `X` and `Y` are independent, but uses predictive information when correlation is present. This is consistent with the theoretical MSE optimum; it is **not evidence of pathological collapse**.

**Why Exp02 follows:** Exp01 has no learned encoder. We next ask whether the same behavior occurs with JEPA-style representations.

## Exp02 — Mean prediction in toy JEPA

**Script:** `experiments/exp02_jepa_collapse.py`

**Why:** Test mean prediction after introducing context and target encoders; compare frozen and EMA target encoders.

| Target mode | rho | Target variance | Prediction variance | Prediction MSE | Mean-baseline MSE |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 0.0 | 0.048764 | 0.000091 | 0.048899 | 0.048793 |
| Frozen | 0.5 | 0.034749 | 0.007633 | 0.027428 | 0.034750 |
| Frozen | 1.0 | 0.038663 | 0.038648 | 0.000007 | 0.038666 |
| EMA | 0.0 | 0.052579 | 0.000045 | 0.052647 | 0.052584 |
| EMA | 0.5 | 0.011904 | 0.002403 | 0.009386 | 0.011910 |
| EMA | 1.0 | 0.589432 | 0.589505 | 0.000010 | 0.589680 |

**What we learned:** At `rho = 0`, predictor outputs are nearly constant even though target representations have nonzero variance. At `rho = 0.5` and `1`, predictions outperform the mean baseline. This separates **predictor mean prediction** from the special case of a **constant target encoder**.

**What we did not learn:** The results do not show that JEPA fails to use available predictive information.

**Why Exp03 follows:** An EMA target encoder changes over time. We therefore examine whether its representation variance shrinks during training.

## Exp03 — Target representation variance dynamics

**Script:** `experiments/exp03_collapse_dynamics.py`

**Why:** Exp02 reports final variance only. Exp03 tracks how target representation variance evolves during optimization.

Define the average variance across the `d` representation dimensions and its initial-normalized ratio:

$$
V(t)=\frac{1}{d}\sum_{j=1}^{d}\mathrm{Var}(z_{y,j}^{(t)}),
\qquad R(t)=\frac{V(t)}{V(0)}.
$$

| rho | Target mode | Initial variance | Final variance | Final ratio | Final prediction MSE |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 0.0 | Frozen | 0.060440 | 0.060440 | 1.000000 | 0.060670 |
| 0.0 | EMA | 0.060440 | 0.034432 | 0.569691 | 0.034534 |
| 0.5 | Frozen | 0.061784 | 0.061784 | 1.000000 | 0.047969 |
| 0.5 | EMA | 0.061784 | 0.063460 | 1.027126 | 0.048912 |
| 1.0 | Frozen | 0.064746 | 0.064746 | 1.000000 | 0.000016 |
| 1.0 | EMA | 0.064746 | 0.443118 | 6.843946 | 0.000025 |

**What we learned:** Frozen target variance stays constant, as expected. Under EMA, variance falls by approximately **43%** at `rho = 0`, remains close to its initial value at `rho = 0.5`, and increases substantially at `rho = 1`.

**What we did not learn:** Lower variance is **not sufficient** to establish information loss. For example, multiplying a nonconstant embedding by `0.5` reduces its variance to one-quarter while preserving its recoverability in exact arithmetic.

**Why Exp04 follows:** We need a separate measure of whether target representations still distinguish their inputs.

## Exp04 — Variance reduction vs. linear recoverability

**Script:** `experiments/exp04_representation_probe.py`

**Why:** Test whether the variance reduction observed in Exp03 is accompanied by worse recovery of the target encoder's input `Y`.

Freeze the encoder, fit a linear probe on training embeddings, and evaluate on held-out data:

$$
\hat Y=Wz_y+b, \qquad \mathcal L_{\mathrm{probe}}=\mathbb E[(\hat Y-Y)^2].
$$

Initial and final encoders are evaluated separately. Probe features are standardized using training-set statistics.

| rho | Target mode | Variance ratio | Initial probe MSE | Final probe MSE | Final probe R² |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 0.0 | Frozen | 1.000000 | 0.000021 | 0.000021 | 0.999978 |
| 0.0 | EMA | 0.569691 | 0.000021 | 0.000003 | 0.999997 |
| 0.5 | Frozen | 1.000000 | 0.000021 | 0.000021 | 0.999979 |
| 0.5 | EMA | 1.027126 | 0.000021 | 0.000007 | 0.999993 |
| 1.0 | Frozen | 1.000000 | 0.000022 | 0.000022 | 0.999979 |
| 1.0 | EMA | 6.843946 | 0.000022 | 0.000002 | 0.999998 |

**What we learned:** For `rho = 0` under EMA, representation variance decreases by approximately **43%**, yet linear-probe MSE improves from about `2.1e-5` to `3e-6`. Across all tested conditions, the probe recovers `Y` with `R²` near `1`.

**Interpretation:** In this one-dimensional toy setup, lower representation variance is **not accompanied by reduced linear recoverability**. The observation is consistent with changes in scale or geometry rather than information collapse.

**Important limitation:** A linear probe measures only *linearly decodable* information. It cannot prove full information preservation. Moreover, the initial probe already performs almost perfectly, making this a relatively easy test.

## Exp05 — Nontrivial mean prediction

**Script:** `experiments/exp05_nontrivial_mean_prediction.py`

**Why:** In Exp01–04, mean-like predictions arose when context and target were independent, so they were consistent with the MSE-optimal conditional mean. Exp05 tests whether the predictor still defaults to the mean when useful predictive information exists.

Generate a nonlinear, deterministic pair:

$$
X\sim\mathcal N(0,1),\qquad Y=X^2-1.
$$

Here, the unconditional mean of `Y` is zero, but `Y` is fully determined by `X`. The theoretical direct-regression mean baseline MSE is `Var(Y) = 2`, while a perfect nonlinear predictor has zero MSE.

Compare a direct MLP trained to predict `Y` against toy JEPA models with frozen or EMA target encoders. The JEPA models predict **target embeddings**, not raw `Y`, so their absolute MSEs are **not directly comparable** to the direct MLP or across encoder modes. Each model is evaluated against its own mean baseline in the same output space.

### Direct MLP results

| Seed | Prediction MSE | Mean-baseline MSE |
| ---: | ---: | ---: |
| 42 | 0.000127 | 1.897328 |
| 123 | 0.000138 | 2.195834 |
| 456 | 0.000286 | 2.054868 |

### Toy JEPA results

| Seed | Target mode | Prediction MSE | Mean-baseline MSE | Prediction variance | Target variance |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 42 | Frozen | 0.000044 | 0.050961 | 0.050952 | 0.050961 |
| 42 | EMA | 0.000137 | 1.415847 | 1.419923 | 1.415847 |
| 123 | Frozen | 0.000039 | 0.055249 | 0.055346 | 0.055249 |
| 123 | EMA | 0.000079 | 1.258154 | 1.259657 | 1.258154 |
| 456 | Frozen | 0.000041 | 0.061238 | 0.061204 | 0.061238 |
| 456 | EMA | 0.000105 | 2.243714 | 2.241472 | 2.243714 |

Relative improvement over a mean baseline is defined as:

$$
I = 1-\frac{L_{\mathrm{pred}}}{L_{\mathrm{mean}}}.
$$

All toy JEPA runs achieve more than **99.9%** relative improvement over their respective mean baselines (using the reported rounded values). Their prediction variances also closely match their target variances, and prediction errors are small. Thus, nearly constant prediction was **not observed** in this predictable nonlinear setting.

**Caveats:** The reported mean baselines use the evaluation-set mean; this is a descriptive reference, not a strictly train-estimated held-out baseline. The target encoder may change its output scale under EMA, so raw loss values alone cannot establish representation quality or superiority of one encoder mode. The result is limited to this one-dimensional synthetic task and three seeds.

**What we learned:** A simplified JEPA can exploit nonlinear predictive information under the tested conditions. This does **not** establish that real JEPA systems are universally free from pathological collapse.

## Overall findings

| Experiment | Research question | Supported finding |
| :--- | :--- | :--- |
| Exp01 | When is mean prediction MSE-optimal? | With independent context and target, a constant conditional mean is optimal. |
| Exp02 | Does toy JEPA exhibit mean-like predictions? | Yes, at `rho = 0`, even with nonconstant target embeddings. |
| Exp03 | How does target representation variance evolve? | EMA variance decreased at `rho = 0` and increased at `rho = 1`. |
| Exp04 | Does variance reduction necessarily reduce recoverability? | No reduction in linear recovery was observed in these toy runs. |
| Exp05 | Does toy JEPA ignore usable nonlinear predictive information? | No: all tested runs substantially beat the mean baseline. |

### Final conclusion

**No evidence of pathological predictor mean collapse was found in the tested toy JEPA configurations.** Mean-like prediction when context and target are independent is consistent with the MSE optimum. When a deterministic nonlinear relation was introduced, the toy JEPA learned to predict the target representation accurately.

This is a **limited negative result**, not a proof that JEPA cannot collapse. In particular, **predictor mean prediction**, **target-encoder representation collapse**, and **information loss** are different phenomena and should not be conflated.

### Scope and limitations

- Synthetic **one-dimensional** inputs and simplified JEPA networks, not full I-JEPA.
- No realistic masking, spatial structure, temporal sequences, or state-tracking tasks.
- Exp01–04 are limited runs; Exp05 uses three random seeds.
- Exp04 tests **linear decodability** of a simple input, not complete information preservation.
- Variance changes can reflect scaling rather than loss of predictive content.
- Exp05's baseline uses evaluation-set means; stronger future evaluations should estimate baselines from training data.
- These experiments do not test whether an explicit scratchpad benefits JEPA.

## Research status and next direction

**Mean-prediction investigation: completed for this toy setting (Exp01–05).**

**Next question:** Can a JEPA-style model track and update hidden states across time when the current observation is insufficient on its own? A future temporal state-tracking experiment should compare a memoryless baseline, a history-aware model, and—only after establishing a clear need—a scratchpad-augmented model. This is a proposed next step, **not a result of Exp01–05**.

## Reproduce the experiments

From the repository root, activate the project's virtual environment and run:

```bash
python experiments/exp01_mean_prediction.py
python experiments/exp02_jepa_collapse.py
python experiments/exp03_collapse_dynamics.py
python experiments/exp04_representation_probe.py
python experiments/exp05_nontrivial_mean_prediction.py
```

**Markdown compatibility:** Equations use GitHub math blocks (`$$`) and supported math commands. Tables use standard GitHub Flavored Markdown.
