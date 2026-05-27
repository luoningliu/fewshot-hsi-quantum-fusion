# Hybrid Quantum-Classical Neural Networks for Hyperspectral Remote Sensing Image Classification

## Current Project Line

This project studies whether hybrid quantum-classical neural networks provide reproducible, explainable, and measurable marginal gains for hyperspectral remote sensing image classification.

The first-round datasets are:

```text
Indian Pines
Pavia University
Salinas
```

The main experimental path is:

```text
HSI cube + ground truth
-> full-image non-background normalization
-> PCA=30
-> 9x9 patch extraction
-> HybridSN-style spectral-spatial encoder
-> Linear / MLP / classical bottleneck / QNN heads
-> OA, AA, Kappa, Macro-F1, per-class recall
```

## Stage Roadmap

1. Stage 1: fixed data protocol for Indian Pines, Pavia University, and Salinas.
2. Stage 2: SVM-RBF, Random Forest, and kNN traditional baselines.
3. Stage 3: 1D-CNN, 2D-CNN, 3D-CNN, and HybridSN-style baselines.
4. Stage 4: HybridSN encoder with Linear, MLP, bottleneck, and QNN classifiers.
5. Stage 5: QNN ablations: random, frozen, no-entanglement, residual, and data re-uploading.
6. Stage 6: quantum structure sweep over qubits, layers, encoding, and entanglement.
7. Stage 7: multi-seed stability.
8. Stage 8: feature-space and confusion analysis.
9. Stage 9: complexity and resource analysis.
10. Stage 10: final report generation.

## Primary Question

The original key question was:

```text
Does HybridSN + QNN outperform HybridSN + MLP under a fixed and fair data protocol?
```

The current evidence has shifted the main research question to:

```text
Does a Spectral QNN branch, placed on the center-pixel spectral input and combined
with metric-learning objectives, provide reproducible marginal-to-moderate gains
in few-shot HSI classification?
```

Current strongest line:

```text
Spectral QNN Gated Fusion + Prototype Loss / SupCon Loss
```

Current few-shot protocol:

```text
datasets = Indian Pines, Pavia University, Salinas
shots = 5, 10
seeds = 0, 1, 2, 3, 4
baseline = HybridSN-small
metrics = OA, AA, Kappa, Macro-F1, Weighted-F1, per-class precision/recall/F1
```

Latest evidence package:

```text
result/final_evidence_closure_fewshot_spectral_qnn_20260526_094921/
```

Current interpretation:

1. Pavia University 5/10-shot is the strongest positive evidence.
2. Indian Pines 5/10-shot shows marginal but reproducible gains.
3. Salinas 5-shot is positive, but Prototype and SupCon differ in stability.
4. Salinas 10-shot negative transfer is mainly tied to Prototype Loss.
5. SupCon Loss mitigates Salinas 10-shot Macro-F1 degradation, but OA and Weighted-F1 remain below HybridSN-small.
6. The QNN branch is better framed as spectral-side decision-boundary regularization, not as a faster or universally stronger classifier head.

## Next Optimization Plan: Toward All-Dataset 5/10-shot Improvement

Goal:

```text
Build a Spectral QNN variant that exceeds HybridSN-small on Indian Pines,
Pavia University, and Salinas for both 5-shot and 10-shot settings.
```

This stage should avoid broad, uncontrolled hyperparameter search. Each change should target the observed failure mode:

```text
Salinas 10-shot is near-saturated for the classical baseline, and the QNN residual
branch can disturb already-good decision boundaries.
```

### Direction 1: Data Re-uploading Spectral QNN

Motivation:

HSI center-pixel spectra are continuous one-dimensional spectral signals. A single spectral encoding layer may only expose low-order spectral responses. Data re-uploading can increase the spectral nonlinearity available to the QNN without simply increasing qubit count.

Candidate variants:

```text
QNN-Reupload-A: q6_l2, full-spectrum re-uploading
QNN-Reupload-B: q6_l2, grouped-band re-uploading
QNN-Reupload-C: shared q6 circuit over spectral groups, pooled output
```

Controlled comparison:

```text
baseline model = Spectral QNN Gated Fusion + SupCon
change only the QNN encoding / re-uploading structure
same splits, same seeds, same PCA=30, same patch size, same optimizer family
```

Priority validation:

```text
Salinas 10-shot seeds 0-4
Pavia University 10-shot seeds 0-4
```

Success criterion:

```text
Salinas 10-shot Macro-F1, OA, and Weighted-F1 no longer fall below HybridSN-small,
while Pavia 10-shot remains positive.
```

Initial implementation status:

```text
Implemented: QNN-Reupload-A
variant = q6_l2 data re-uploading + multi-observable readout
loss = SupCon
minimal batch = Salinas 10-shot seeds 0-4, Pavia University 10-shot seeds 0-4
result_dir = result/qnn_reupload_supcon_minibatch_salinas_pavia_10shot_20260526_103205/
```

Initial result:

```text
Salinas 10-shot:
  Delta vs HybridSN-small:
    OA = -0.0226
    Macro-F1 = +0.0025
    Weighted-F1 = -0.0229

Pavia University 10-shot:
  Delta vs HybridSN-small:
    OA = +0.0474
    Macro-F1 = +0.0799
    Weighted-F1 = +0.0490
```

Decision:

```text
QNN-Reupload-A does not pass the acceptance rule because Salinas 10-shot still
underperforms HybridSN-small on OA and Weighted-F1. Keep this as a diagnosed
negative/partial variant, not the next mainline model.
```

Next action for this direction:

```text
Do not expand QNN-Reupload-A to the full 5/10-shot matrix.
Only revisit data re-uploading after adding residual-safe scaling or
confidence-aware gate protection.
```

### Direction 2: Residual-Safe / Identity-Initialized QNN Branch

Motivation:

The current gated QNN branch can perturb a strong classical boundary, especially on Salinas 10-shot. A safer residual formulation should let the QNN start near no-op and only contribute when useful.

Candidate implementation:

```text
scale = sigmoid(alpha)
logits = base_logits + scale * gate * qnn_logits
alpha initialized to a negative value, e.g. -4.0
```

Alternative schedule:

```text
alpha warmup: 0 -> 1 over the first 20 epochs
```

Candidate variants:

```text
QNN-ResidualSafe-A: learnable alpha initialized near zero
QNN-ResidualSafe-B: alpha warmup schedule
QNN-ResidualSafe-C: classwise alpha gate
```

Priority validation:

```text
Salinas 10-shot seeds 0-4
Salinas 5-shot seeds 0-4
Pavia University 10-shot seeds 0-4
```

Success criterion:

```text
Reduce Salinas 10-shot OA / Weighted-F1 degradation without losing Pavia gains.
```

Initial implementation status:

```text
Implemented: QNN-ResidualSafe-A
variant = standard q6_l1 Spectral QNN + SupCon + learnable residual scale
residual_scale = sigmoid(alpha), alpha_init = -4.0
minimal batch = Salinas 10-shot seeds 0-4, Pavia University 10-shot seeds 0-4
result_dir = result/qnn_residualsafe_supcon_minibatch_salinas_pavia_10shot_20260527_001133/
```

Initial result:

```text
Salinas 10-shot:
  Delta vs HybridSN-small:
    OA = -0.0128
    Macro-F1 = +0.0016
    Weighted-F1 = -0.0129
  Delta vs original SupCon QNN:
    OA = +0.0061
    Macro-F1 = -0.0010
    Weighted-F1 = +0.0072

Pavia University 10-shot:
  Delta vs HybridSN-small:
    OA = +0.0221
    Macro-F1 = +0.0420
    Weighted-F1 = +0.0253
  Delta vs original SupCon QNN:
    OA = -0.0188
    Macro-F1 = -0.0320
    Weighted-F1 = -0.0170
```

Decision:

```text
QNN-ResidualSafe-A partially reduces Salinas 10-shot negative transfer compared
with the original SupCon QNN, but it still does not exceed HybridSN-small on
Salinas OA / Weighted-F1. It also sacrifices a large part of the Pavia 10-shot
gain, suggesting that alpha_init=-4.0 is too conservative for settings where
the QNN branch is useful.
```

Next action for this direction:

```text
Do not expand QNN-ResidualSafe-A to the full 5/10-shot matrix.
Try QNN-ResidualSafe-B with alpha warmup or a less restrictive initialization
such as alpha_init=-2.0, and keep Salinas 10-shot + Pavia 10-shot as the
acceptance gate before full expansion.
```

Follow-up implementation status:

```text
Implemented: QNN-ResidualSafe-B
variant = standard q6_l1 Spectral QNN + SupCon + learnable residual scale
residual_scale = warmup_factor * sigmoid(alpha)
alpha_init = -2.0
residual_warmup_epochs = 20
minimal batch = Salinas 10-shot seeds 0-4, Pavia University 10-shot seeds 0-4
result_dir = result/qnn_residualsafe_b_supcon_minibatch_salinas_pavia_10shot_20260527_102113/
```

Follow-up result:

```text
Salinas 10-shot:
  Delta vs HybridSN-small:
    OA = -0.0218
    Macro-F1 = -0.0032
    Weighted-F1 = -0.0235
  Delta vs original SupCon QNN:
    OA = -0.0030
    Macro-F1 = -0.0058
    Weighted-F1 = -0.0034
  Delta vs ResidualSafe-A:
    OA = -0.0090
    Macro-F1 = -0.0047
    Weighted-F1 = -0.0106

Pavia University 10-shot:
  Delta vs HybridSN-small:
    OA = +0.0217
    Macro-F1 = +0.0370
    Weighted-F1 = +0.0242
  Delta vs original SupCon QNN:
    OA = -0.0192
    Macro-F1 = -0.0369
    Weighted-F1 = -0.0181
  Delta vs ResidualSafe-A:
    OA = -0.0004
    Macro-F1 = -0.0049
    Weighted-F1 = -0.0011
```

Decision after ResidualSafe-B:

```text
QNN-ResidualSafe-B fails the Salinas 10-shot acceptance gate and does not
recover the original Pavia 10-shot SupCon-QNN gain. Relaxing alpha and adding
warmup increases the QNN residual scale, but it amplifies unstable Salinas
seed behavior instead of solving negative transfer.
```

Next action after ResidualSafe-B:

```text
Stop the global residual-scale line for now. Move to a more conditional
mechanism: either confidence-aware quantum gate / negative-transfer guard, or
Direction 3 Multi-Prototype / Class-Conditional Quantum Metric Branch.
Given Salinas class diversity, prioritize Direction 3 unless a lightweight
confidence guard can be added without changing the data protocol.
```

### Direction 3: Multi-Prototype / Class-Conditional Quantum Metric Branch

Motivation:

Salinas contains large, internally diverse classes such as Vinyard_untrained and Grapes_untrained. A single class prototype can over-compress class structure and amplify confusion. Multi-prototype metric heads may better represent multi-modal spectral distributions.

Candidate variants:

```text
QNN-MultiProto-A: 2 prototypes per class
QNN-MultiProto-B: 3 prototypes per class
QNN-MultiProto-C: softmin over class prototypes
QNN-MultiProto-D: class-conditional prototype temperature
```

Prototype construction:

```text
support-only class means
or k-means on training/support fused features
```

Priority validation:

```text
Salinas 10-shot seeds 0-4
Salinas 5-shot seeds 0-4
```

Success criterion:

```text
Recover F1 for Vinyard_untrained and Grapes_untrained while preserving gains on smaller classes.
```

Initial implementation status:

```text
Implemented: QNN-MultiProto-A
variant = standard q6_l1 Spectral QNN + Prototype Loss
prototypes_per_class = 2
prototype aggregation = deterministic support split + logsumexp over sub-prototype distances
minimal batch = Salinas 10-shot seeds 0-4, Pavia University 10-shot seeds 0-4
result_dir = result/qnn_multiproto2_minibatch_salinas_pavia_10shot_20260527_113147/
```

Initial result:

```text
Salinas 10-shot:
  Delta vs HybridSN-small:
    OA = -0.0294
    Macro-F1 = -0.0051
    Weighted-F1 = -0.0340
  Delta vs original Prototype QNN:
    OA = -0.0006
    Macro-F1 = -0.0003
    Weighted-F1 = -0.0007
  Delta vs original SupCon QNN:
    OA = -0.0105
    Macro-F1 = -0.0077
    Weighted-F1 = -0.0138

Pavia University 10-shot:
  Delta vs HybridSN-small:
    OA = +0.0405
    Macro-F1 = +0.0692
    Weighted-F1 = +0.0412
  Delta vs original Prototype QNN:
    OA = -0.0001
    Macro-F1 = -0.0001
    Weighted-F1 = -0.0001
```

Decision:

```text
QNN-MultiProto-A does not pass the Salinas 10-shot acceptance gate. It nearly
matches the original Prototype-QNN line, strengthens the existing Pavia-positive
story, but does not solve the target negative case. The deterministic
sub-prototype split is not sufficient for Salinas class ambiguity.
```

Next action for this direction:

```text
Do not expand QNN-MultiProto-A to the full matrix. If continuing Direction 3,
use a safer class-conditional mechanism: class-conditional prototype
temperature, per-class metric weight, or confidence-gated metric loss. Given
the repeated Salinas seed0/seed4 degradation, Direction 5 confidence-aware
negative-transfer guard is now the higher-priority next experiment.
```

### Direction 4: QCNN-style Hierarchical Spectral Circuit

Motivation:

The current spectral QNN treats the PCA spectrum as a compact vector. A hierarchical local spectral circuit may work more like a learnable spectral filter and reduce over-global mixing.

Candidate structure:

```text
30 PCA bands -> 5 groups x 6 bands
shared q6 local circuit per group
group features -> attention / mean pooling / gated pooling
pooled quantum spectral feature -> gated fusion logits
```

Candidate variants:

```text
QNN-QCNN-A: shared local q6 circuit + mean pooling
QNN-QCNN-B: shared local q6 circuit + attention pooling
QNN-QCNN-C: two-level local-to-global quantum pooling
```

Priority validation:

```text
Pavia University 5/10-shot
Salinas 10-shot
```

Success criterion:

```text
Maintain Pavia gains and reduce Salinas 10-shot class confusion.
```

### Direction 5: Confidence-Aware Quantum Gate / Negative-Transfer Guard

Motivation:

When HybridSN-small is already highly confident, especially on Salinas 10-shot, the QNN branch should not strongly modify the decision unless the sample is near a boundary.

Candidate gate inputs:

```text
z
center spectrum
base max softmax confidence
base top1-top2 logit margin
```

Candidate regularization:

```text
gate_penalty = mean(gate * stopgrad(base_confidence))
```

or:

```text
if base_margin is high, penalize large QNN residual contribution
if base_margin is low, allow QNN contribution
```

Candidate variants:

```text
QNN-ConfGate-A: add confidence and logit margin to gate input
QNN-ConfGate-B: confidence-weighted gate penalty
QNN-ConfGate-C: residual norm penalty on high-confidence samples
```

Priority validation:

```text
Salinas 10-shot seeds 0-4
Pavia University 10-shot seeds 0-4
Indian Pines 10-shot seeds 0-4
```

Success criterion:

```text
Salinas 10-shot OA and Weighted-F1 improve over HybridSN-small, while Macro-F1
does not degrade on Indian Pines and Pavia University.
```

## Recommended Execution Order

Run the next experiments in this order:

```text
1. QNN-ResidualSafe + SupCon
2. QNN-ConfGate + SupCon
3. QNN-Reupload + SupCon
4. QNN-MultiProto on Salinas-focused diagnostics
5. QNN-QCNN if the first three directions still fail on Salinas 10-shot
```

Minimal first batch:

```text
Salinas 10-shot, seeds 0-4
Pavia University 10-shot, seeds 0-4
```

Only expand to the full matrix if the minimal batch is positive:

```text
Indian Pines 5/10-shot
Pavia University 5/10-shot
Salinas 5/10-shot
seeds 0-4
```

Decision rule:

```text
Do not accept a new QNN variant unless it improves Salinas 10-shot without
sacrificing Pavia University 5/10-shot.
```

Final target:

```text
For every dataset-shot pair in {Indian Pines, Pavia University, Salinas} x {5, 10},
QNN variant mean Macro-F1, OA, and Weighted-F1 should be >= HybridSN-small.
```
