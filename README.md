# Mean Prediction and Representation Dynamics in Toy JEPA

This repository investigates **mean prediction**, **representation variance**, and **linear recoverability** in a simplified Joint-Embedding Predictive Architecture (JEPA). It does **not** implement full I-JEPA, and the experiments do not establish pathological collapse in JEPA generally.

## Research questions

1. When does an MSE predictor output a nearly constant (mean) prediction?
2. Does the same behavior occur in a toy JEPA with encoders?
3. How does target-encoder representation variance change during training?
4. Does a decrease in representation variance coincide with a loss of linearly recoverable input information?

## Common setup

Generate independent standard-normal variables `X` and `epsilon`, then define:

$$
Y = \rho X + \sqrt{1-\rho^2}\,\epsilon,
\qquad \rho \in \{0, 0.5, 1\}.
$$

- `rho = 0`: `X` and `Y` are independent.
- `rho = 0.5`: `X` provides partial information about `Y`.
- `rho = 1`: `Y = X`.

For Exp02–04, the toy JEPA consists of a context encoder, target encoder, and predictor:

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
V(t)=\frac{1}{d}\sum_{j=1}^{d}\operatorname{Var}(z_{y,j}^{(t)}),
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

## Overall findings

| Experiment | Question | Supported conclusion |
| :--- | :--- | :--- |
| Exp01 | When does MSE regression predict the mean? | When context lacks predictive information, mean prediction is optimal. |
| Exp02 | Does mean prediction also occur in toy JEPA? | Yes, at `rho = 0`, despite nonconstant target embeddings. |
| Exp03 | Does target representation variance change during training? | Yes, EMA variance changes differently across correlation conditions. |
| Exp04 | Does reduced variance imply lost recoverability? | No decrease in linear-probe recovery was observed in these runs. |

### What these experiments do **not** establish

- Pathological mean collapse in actual I-JEPA or other JEPA models.
- That EMA necessarily causes representation collapse.
- That reduced variance causes mean prediction.
- That all input information is preserved.
- That a scratchpad would improve JEPA.

### Limitations

- One-dimensional synthetic Gaussian data, not realistic images or temporal sequences.
- Simplified JEPA, without the full I-JEPA masking/position-conditioning setup.
- One random seed per condition; replication is still needed.
- Exp04's linear reconstruction task is nearly solved even before training.

## Next step — Exp05: Nontrivial mean prediction

**Question:** Can the toy JEPA produce nearly constant predictions **even when its context contains exploitable predictive information**?

Design a predictable but nontrivial context-target relationship and compare mean baselines, a capable predictor, and toy JEPA under matched evaluation targets. If the JEPA remains at the mean baseline while a comparable model performs better, investigate optimization, architecture, and encoder behavior before attributing the failure to JEPA generally.

The longer-term research direction is **temporal state tracking** and whether an explicit **scratchpad** helps, but neither has been evaluated in Exp01–04.

## Run experiments

From the repository root, with the project's virtual environment activated:

```bash
python experiments/exp01_mean_prediction.py
python experiments/exp02_jepa_collapse.py
python experiments/exp03_collapse_dynamics.py
python experiments/exp04_representation_probe.py
```

**Note:** Mathematical expressions use GitHub-supported `$$ ... $$` Markdown math blocks. For correct rendering, view `README.md` on GitHub rather than in a plain-text terminal.
