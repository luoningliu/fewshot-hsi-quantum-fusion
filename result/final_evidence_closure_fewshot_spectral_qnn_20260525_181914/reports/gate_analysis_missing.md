# Gate Analysis Missing

No saved gate-value CSV files were found in the current result tree.

Missing files: `*_gate_values.csv` for Spectral QNN Gated Fusion evaluations.

The script that already contains gate export support is `scripts/run_fair_control_models_fewshot.py`, via `_write_gate_values()`.
Future runs should call the model with `return_aux=True` during evaluation and save per-sample `mean_gate`, `gate_for_pred_class`, and `gate_for_true_class`.
