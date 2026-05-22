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

The first key question is:

```text
Does HybridSN + QNN outperform HybridSN + MLP under a fixed and fair data protocol?
```

If this does not hold, the project should shift toward careful negative-result analysis rather than a large quantum sweep.

