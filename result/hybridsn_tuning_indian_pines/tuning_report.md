# HybridSN Tuning Report: Indian Pines

## Architecture

HybridSN uses 3D convolution blocks for joint spectral-spatial features, reshapes `[B, C, D, H, W]` into `[B, C*D, H, W]`, then applies a 2D convolution block and an MLP classifier.

## Input Shape

- Dataset patch: `[B, patch_size, patch_size, pca_components]`.
- Model input: `[B, 1, pca_components, patch_size, patch_size]`.
- Background label 0 is removed; labels are remapped to `0..15`.

## Search Space Executed

- Stage A attempted configs: 8.
- Stage B attempted configs: 9.
- Stage C confirmation seeds: [0, 1].
- Selection metric: validation Macro-F1, with validation AA as tie-break context.

## Best Config

```yaml
architecture: hybridsn_base
patch_size: 11
pca_components: 30
learning_rate: 0.001
batch_size: 16
weight_decay: 0.0
dropout: 0.5
loss_name: weighted_ce
class_weight_mode: inverse_sqrt
embedding_dim: 128
seed: 0
```

## Test Metrics

| Model | OA | AA | Kappa | Macro-F1 | Weighted-F1 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Fixed 1D-CNN | 59.28 | 46.99 | 52.97 | 46.33 | 56.63 | already fixed |
| Fixed 3D-CNN | 63.68 | 49.08 | 57.56 | 49.00 | 59.92 | already fixed |
| HybridSN initial | 81.00 | 68.69 | 78.27 | 69.97 | 80.48 | before tuning |
| HybridSN tuned | 98.80 | 97.27 | 98.64 | 97.19 | 98.81 | best config |

## Assessment

- Clear improvement over fixed 3D-CNN: yes.
- Meets good target (`OA >= 80`, `Macro-F1 >= 70`, `AA >= 70`): yes.
- The tuned HybridSN is suitable as the main classical encoder if final multi-seed test metrics remain stable and no QNN-specific tuning is mixed into this baseline.

## Remaining Issues

- This runner supports a limited staged search budget to keep CPU runs tractable; increase `stage_a_limit`, `stage_b_limit`, and `confirmation_seeds` for paper-grade tuning.
- Classification map is generated on the test split only; full-image inference for every non-background pixel can be added if needed.
- The split is a random pixel-level split. Patch-based CNNs can benefit from neighboring spatial context across train/test pixels, so these results are valid for the existing project protocol but should not be interpreted as spatially isolated generalization.
