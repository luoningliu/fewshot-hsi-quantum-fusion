# Limitations and Negative Cases

1. QNN does not universally outperform HybridSN-small; gains are dataset- and shot-dependent.
2. Salinas 10-shot is a negative transfer case, likely because the classical baseline is already near saturation.
3. Final-head QNN does not outperform MLP head in fair frozen-embedding comparisons.
4. Random pixel split can be optimistic for patch-based HSI models because neighboring pixels may share spatial context.
5. Spatial split remains difficult; current spatial pilots show large performance drops.
6. QNN simulation is slower than classical heads under the current CPU/classical simulator setup.
7. Prototype geometry does not universally improve; logit margin gives a more consistent explanation for successful settings.
