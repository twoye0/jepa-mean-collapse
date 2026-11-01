Investigating Mean Prediction and Representation Collapse in Toy JEPA

1. Research Objective

This project investigates mean prediction and representation collapse in a simplified Joint-Embedding Predictive Architecture (JEPA).

The central research question is:

Does JEPA exhibit pathological mean-prediction collapse, or does mean prediction simply reflect the optimal behavior of an MSE-based predictor when predictive information is unavailable?

We conduct four preliminary experiments to distinguish:

1. Mean Prediction: The predictor produces nearly constant outputs.
2. Representation Collapse: The encoder maps different inputs to identical or nearly identical representations.
3. Information Loss: The learned representations lose information about their original inputs.

Importantly, these phenomena are related but not equivalent.

The experiments use a simplified JEPA architecture rather than a complete I-JEPA implementation. Consequently, the findings should not be generalized directly to JEPA models in realistic settings.

⸻

2. Experimental Setup

2.1 Synthetic Dataset

We construct correlated Gaussian variables:

[
X,\epsilon\sim\mathcal N(0,1), \qquad X\perp\epsilon
]

[
Y=\rho X+\sqrt{1-\rho^2}\epsilon
]

where (\rho\in{0,0.5,1}) controls the correlation between the context (X) and target (Y).

Correlation	Interpretation
(\rho=0)	Context and target are independent
(\rho=0.5)	Context contains partial predictive information
(\rho=1)	Target is completely determined by context

2.2 Toy JEPA Architecture

The model consists of three components:

* Context Encoder: (z_x=f_\theta(X))
* Target Encoder: (z_y=f_{\bar\theta}(Y))
* Predictor: (\hat z_y=g_\phi(z_x))

The training objective is:

[
\mathcal L=\mathbb E[|\hat z_y-z_y|_2^2]
]

We compare two target encoder configurations.

Frozen Target Encoder: Target encoder parameters remain fixed throughout training.

EMA Target Encoder: Target encoder parameters are updated using an exponential moving average of the context encoder:

[
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta
]

with (\tau=0.99).

⸻

3. Experiment 01 — Mean Prediction under MSE

File: experiments/exp01_mean_prediction.py

Research Question

Does an MSE-based predictor converge to the target mean when the context contains no predictive information?

Motivation

For squared-error regression, the optimal predictor is the conditional expectation:

[
g^*(X)=\mathbb E[Y\mid X]
]

For our Gaussian construction:

[
\mathbb E[Y\mid X]=\rho X
]

Therefore, when (\rho=0), the optimal prediction is the constant target mean.

This experiment verifies the theoretical behavior before introducing trainable encoders.

Results

(\rho)	Prediction MSE	Prediction Variance	Mean Baseline MSE
0.0	1.0269	0.0044	1.0223
0.5	0.7707	0.2462	1.0139
1.0	0.0000	0.9962	0.9962

Interpretation

* (\rho=0): The predictor produces nearly constant outputs because the context provides no information about the target.
* (\rho=0.5): The predictor exploits partial predictive information and outperforms the mean baseline.
* (\rho=1): The predictor accurately reconstructs the target.

Conclusion

Mean prediction can be the mathematically optimal behavior of an MSE predictor.

Near-constant predictions alone do not establish a learning failure.

⸻

4. Experiment 02 — Mean Prediction in Toy JEPA

File: experiments/exp02_jepa_collapse.py

Research Question

Does mean prediction also occur when the model includes context and target encoders?

Motivation

Exp01 considered a direct predictor without encoders.

JEPA introduces learned representations, potentially creating additional mechanisms for degenerate solutions.

Exp02 examines whether mean prediction persists in a simplified JEPA architecture under frozen and EMA target encoder configurations.

Results

Mode	(\rho)	Target Variance	Prediction Variance	Prediction MSE
Frozen	0.0	0.048764	0.000091	0.048899
Frozen	0.5	0.034749	0.007633	0.027428
Frozen	1.0	0.038663	0.038648	0.000007
EMA	0.0	0.052579	0.000045	0.052647
EMA	0.5	0.011904	0.002403	0.009386
EMA	1.0	0.589432	0.589505	0.000010

Interpretation

1. Mean prediction without constant target representations

When (\rho=0), prediction variance approaches zero while target representation variance remains positive.

This indicates that nearly constant predictor outputs can occur even when the target encoder produces nonconstant representations.

2. Predictive information is utilized when available

For (\rho=0.5), prediction MSE is lower than the corresponding mean baseline.

For (\rho=1), prediction MSE approaches zero.

Thus, the toy JEPA predictor does not universally converge to constant outputs.

Conclusion

Mean prediction occurs in the toy JEPA architecture when context and target are independent.

However, this behavior is consistent with optimal conditional-mean prediction and does not demonstrate pathological mean collapse.

Follow-up Question

Because the EMA target encoder changes during training, could its representations become progressively less distinguishable?

This motivates Exp03.

⸻

5. Experiment 03 — Representation Variance Dynamics

File: experiments/exp03_collapse_dynamics.py

Research Question

How does target representation variance evolve during training?

Motivation

Exp02 measures representation variance after training but does not explain its evolution.

Exp03 tracks target representation variance throughout optimization to determine whether the encoder representations become more or less dispersed.

Metric

We measure the average variance across representation dimensions:

[
V(t)=\frac{1}{d}\sum_{j=1}^{d}\operatorname{Var}(z_{y,j}^{(t)})
]

and normalize it by the initial variance:

[
R(t)=\frac{V(t)}{V(0)}
]

Results

(\rho)	Mode	Initial Variance	Final Variance	Variance Ratio
0.0	Frozen	0.060440	0.060440	1.000000
0.0	EMA	0.060440	0.034432	0.569691
0.5	Frozen	0.061784	0.061784	1.000000
0.5	EMA	0.061784	0.063460	1.027126
1.0	Frozen	0.064746	0.064746	1.000000
1.0	EMA	0.064746	0.443118	6.843946

Interpretation

Frozen Target Encoder

Representation variance remains unchanged because encoder parameters are fixed.

EMA, (\rho=0)

Representation variance decreases by approximately 43%.

This indicates reduced representation dispersion during training.

EMA, (\rho=0.5)

Representation variance changes non-monotonically and finishes slightly above its initial value.

EMA, (\rho=1)

Representation variance increases substantially while prediction error remains close to zero.

Conclusion

EMA target representation variance exhibits different dynamics depending on context-target correlation.

However:

A reduction in representation variance does not necessarily imply information loss or representation collapse.

For example, scaling an injective representation by a nonzero constant reduces its variance without destroying information.

Follow-up Question

Does the variance reduction observed in Exp03 correspond to a loss of recoverable input information?

This motivates Exp04.

⸻

6. Experiment 04 — Representation Variance vs. Information Preservation

File: experiments/exp04_representation_probe.py

Research Question

Does reduced representation variance imply that the target encoder loses information about its input?

Motivation

Exp03 observed a substantial decrease in target representation variance for the EMA encoder when (\rho=0).

However, variance reduction may result from changes in representation scale rather than loss of distinguishability.

To investigate this, we evaluate whether the original target input can still be reconstructed from its learned representation.

Method

We fit a linear probe:

[
\hat Y=Wz_y+b
]

where (z_y=f_{\bar\theta}(Y)).

The encoder is frozen during probe fitting.

We evaluate reconstruction performance on held-out data using:

* Probe MSE: Reconstruction error.
* Probe (R^2): Fraction of target variance explained by the probe.
* Representation Variance: Dispersion of target embeddings.

Probes are fitted separately to the initial and final encoder representations.

Results

(\rho)	Mode	Variance Ratio	Initial Probe MSE	Final Probe MSE	Final (R^2)
0.0	Frozen	1.000000	0.000021	0.000021	0.999978
0.0	EMA	0.569691	0.000021	0.000003	0.999997
0.5	Frozen	1.000000	0.000021	0.000021	0.999979
0.5	EMA	1.027126	0.000021	0.000007	0.999993
1.0	Frozen	1.000000	0.000022	0.000022	0.999979
1.0	EMA	6.843946	0.000022	0.000002	0.999998

Interpretation

1. Frozen target representations remain unchanged

Both representation variance and linear probe performance remain constant.

2. Variance reduction does not coincide with degraded linear reconstruction

For EMA with (\rho=0):

* Representation variance decreases by approximately 43%.
* Probe MSE decreases from approximately (2.1\times10^{-5}) to (3\times10^{-6}).
* Probe (R^2) remains close to 1.

Therefore, the observed variance reduction is not accompanied by a measurable loss of linearly recoverable input information.

3. Representation variance and probe performance are distinct quantities

Across EMA conditions, representation variance can increase or decrease while linear reconstruction remains highly accurate.

Conclusion

Exp04 provides evidence that the variance reduction observed in Exp03 does not necessarily indicate information collapse.

The results are consistent with changes in representation scale or geometry that preserve linear decodability.

However, linear probing does not measure all forms of information preservation, and the one-dimensional synthetic task is relatively simple.

Therefore, these results do not establish the absence of representation collapse in general JEPA models.

⸻

7. Summary of Findings

Experiment	Main Question	Finding
Exp01	Why does an MSE predictor output the mean?	Mean prediction is optimal when predictive information is absent.
Exp02	Does mean prediction occur in toy JEPA?	Yes, under independent context-target conditions.
Exp03	Does target representation variance change during training?	Yes, EMA variance dynamics depend on context-target correlation.
Exp04	Does variance reduction imply information collapse?	No evidence of degraded linear reconstruction was observed.

Overall Interpretation

Our experiments support three preliminary observations:

1. Mean prediction is not necessarily pathological. It can arise as the optimal solution to squared-error regression.
2. Predictor mean prediction and representation collapse are distinct phenomena. A predictor may produce nearly constant outputs even when target representations remain nonconstant.
3. Representation variance is insufficient to diagnose information collapse. Reduced variance does not necessarily imply reduced input recoverability.

What Has Not Been Established

The experiments do not establish that:

* JEPA generally suffers from pathological mean collapse.
* EMA target encoders necessarily undergo representation collapse.
* Representation variance reduction causes predictor mean collapse.
* Scratchpad mechanisms mitigate these phenomena.

⸻

8. Limitations

* Experiments use one-dimensional synthetic Gaussian data.
* The architecture is a simplified JEPA, not full I-JEPA.
* Experiments were conducted using a single random seed.
* Linear probing measures only linearly recoverable information.
* The experiments do not directly test temporal state tracking.
* No pathological mean-prediction failure has yet been demonstrated.

⸻

9. Next Research Question

Exp05 — Nontrivial Mean Prediction

Research Question

Can a JEPA predictor converge to near-mean predictions even when useful predictive information is available?

The next experiment will investigate whether a toy JEPA model can fail to exploit predictable, potentially nonlinear context-target relationships.

To distinguish a learning failure from an optimal conditional-mean solution, we will compare the model against suitable mean-prediction and predictive baselines.

This experiment is intended to assess whether there is evidence of a nontrivial mean-prediction failure in the toy setting.

A negative result would not establish that such failures are impossible in larger JEPA architectures.

⸻

10. Research Status

Experiment	Status
Exp01 — Mean Prediction	Completed
Exp02 — Toy JEPA Mean Prediction	Completed
Exp03 — Representation Variance Dynamics	Completed
Exp04 — Linear Probe Analysis	Completed
Exp05 — Nontrivial Mean Prediction	Planned
Temporal State Tracking	Not Started
Scratchpad Evaluation	Not Started

Current conclusion: The completed experiments demonstrate expected conditional-mean prediction and correlation-dependent representation dynamics in a simplified JEPA model. They do not provide evidence of pathological mean collapse or substantial loss of linearly recoverable information.