# Mean Collapse in JEPA

Investigating mean prediction and representation collapse in
Joint-Embedding Predictive Architectures (JEPA).

## 1. Research Question

Does JEPA exhibit mean-prediction collapse, and under what conditions?

## 2. Background

JEPA predicts target representations from context representations.

For a simplified predictor with an MSE objective,

$begin:math:display$
\\mathcal\{L\}\(\\phi\)
\=
\\mathbb\{E\}\_\{X\,Y\}
\\left\[
\\\|g\_\\phi\(X\)\-Y\\\|\_2\^2
\\right\]\.
$end:math:display$

The optimal predictor is the conditional expectation:

$begin:math:display$
g\^\*\(X\)\=\\mathbb\{E\}\[Y\\mid X\]\.
$end:math:display$

When the context and target representations are independent,

$begin:math:display$
X\\perp Y\,
$end:math:display$

the optimal prediction becomes

$begin:math:display$
g\^\*\(X\)\=\\mathbb\{E\}\[Y\]\.
$end:math:display$

This motivates investigating when predictors produce
nearly constant, mean-like outputs.

Importantly, mean prediction is not necessarily a training failure.

## 3. Experiments

### Exp01: Mean Prediction with Fixed Representations

**Objective:** Verify the theoretical behavior of an MSE-trained
predictor under different levels of context-target correlation.

Dataset:

$begin:math:display$
X\,\\epsilon\\sim\\mathcal\{N\}\(0\,1\)
$end:math:display$

$begin:math:display$
Y\=\\rho X\+\\sqrt\{1\-\\rho\^2\}\\epsilon
$end:math:display$

where $begin:math:text$\\rho\\in\\\{0\,0\.5\,1\\\}$end:math:text$.

Evaluation:
- Prediction MSE
- Prediction Variance
- Mean Baseline MSE

### Exp02: Mean Collapse in JEPA

**Objective:** Investigate predictor mean collapse and
representation collapse with learnable encoders.

Status: Planned.

## 4. Research Status

- [x] Research question formulated
- [x] Mathematical motivation established
- [x] Exp01 implemented
- [x] Exp01 results analyzed
- [ ] Exp02 implemented

## 5. References

Assran et al., "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture," CVPR 2023.

### Exp01 Results

We investigated mean prediction under three levels of
context-target correlation.

| Correlation (ρ) | Theoretical MSE | Experimental MSE | Theoretical Variance | Experimental Variance |
|---|---|---|---|---|
| 0.0 | 1.0000 | 1.0269 | 0.0000 | 0.0044 |
| 0.5 | 0.7500 | 0.7707 | 0.2500 | 0.2462 |
| 1.0 | 0.0000 | 0.0000 | 1.0000 | 0.9962 |

#### Observations

1. When ρ = 0, the predictor produces nearly constant outputs.
2. When ρ = 0.5, the predictor captures partial predictive information.
3. When ρ = 1, the predictor accurately reconstructs the target.

#### Conclusion

The experimental results are consistent with the theoretical
optimal predictor:

g*(X) = E[Y | X].

When context and target are independent, the optimal
prediction becomes the unconditional target mean.

This experiment validates the expected behavior of MSE
regression but does not establish mean collapse in JEPA.

**Status:** Completed (Sanity Check).