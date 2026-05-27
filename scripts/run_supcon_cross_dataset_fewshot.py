from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from scripts.run_fair_control_models_fewshot import (
    FeatureDataset,
    _load_or_extract_features,
    _load_split,
    _resolve_device,
    _write_gate_values,
)
from scripts.run_hybridsn_small_fewshot import _load_dataset_config, _preprocess_full_image
from scripts.run_hybridsn_small_spectral_qnn_gated_metric_fewshot import SpectralQNNGatedMetricFusion
from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.datasets.hsi_dataset import load_hsi_mat
from src.utils.seed import set_seed


METRICS = ("OA", "AA", "Kappa", "Macro-F1", "Weighted-F1")
MODEL_NAME = "spectral_qnn_gated_supcon"
JSON_MODEL_NAME = "hybridsn_small_spectral_qnn_gated_fusion_supcon"


def main() -> None:
    args = _build_parser().parse_args()
    out = Path(args.output_dir)
    _prepare_output(out)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_training_config(out, args)
    device = _resolve_device(args.device)
    all_rows: list[dict[str, Any]] = _load_existing_rows(out) if args.resume else []
    failures: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        data_cfg = _load_dataset_config(dataset_name, args.data_root)
        raw = load_hsi_mat(data_cfg)
        rows, cols = np.nonzero(raw.gt != raw.background_label)
        labels = raw.gt[rows, cols].astype(np.int64) - 1
        num_classes = int(data_cfg["num_classes"])
        cube_pca, pca_evr_sum = _preprocess_full_image(raw.cube, args.pca_bands, args.seed)
        radius = args.patch_size // 2
        padded_cube = np.pad(cube_pca, ((radius, radius), (radius, radius), (0, 0)), mode="reflect").astype(np.float32)

        for shot in args.shots:
            for seed in args.seeds:
                if args.resume and _completed(out, dataset_name, int(shot), int(seed), args):
                    print(f"[SKIP] completed {dataset_name} SupCon {int(shot)}-shot seed{int(seed)}")
                    continue
                try:
                    row = _run_one(
                        args=args,
                        out=out,
                        dataset_name=dataset_name,
                        data_cfg=data_cfg,
                        padded_cube=padded_cube,
                        rows=rows,
                        cols=cols,
                        labels=labels,
                        num_classes=num_classes,
                        shot=int(shot),
                        seed=int(seed),
                        pca_evr_sum=float(pca_evr_sum),
                        device=device,
                    )
                    all_rows.append(row)
                except Exception as exc:
                    failure = {
                        "dataset": dataset_name,
                        "shot": int(shot),
                        "seed": int(seed),
                        "model": _model_name(args),
                        "stage": "supcon_run",
                        "status": "failed",
                        "reason": repr(exc),
                    }
                    failures.append(failure)
                    print(f"[WARN] failed {failure}")
                _write_outputs(out, all_rows, failures)

    _write_outputs(out, all_rows, failures)
    print(f"Result directory: {out}")


def _run_one(
    args: argparse.Namespace,
    out: Path,
    dataset_name: str,
    data_cfg: dict[str, Any],
    padded_cube: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    shot: int,
    seed: int,
    pca_evr_sum: float,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    split_path = Path(args.split_dir) / f"{dataset_name}_seed{seed}_{shot}shot.json"
    split = _load_split(split_path)
    _copy_cached_features_if_available(args, out, dataset_name, shot, seed)
    z, spectra, feature_path = _load_or_extract_features(
        args, out, dataset_name, padded_cube, rows, cols, labels, num_classes, shot, seed, device
    )
    model = SpectralQNNGatedMetricFusion(
        embedding_dim=z.shape[1],
        spectral_dim=spectra.shape[1],
        num_classes=num_classes,
        gate_mode=args.gate_mode,
        qnn_variant=args.qnn_variant,
        residual_scale_mode=args.residual_scale_mode,
        residual_alpha_init=args.residual_alpha_init,
        qubits=args.qubits,
        layers=args.qnn_layers,
        entanglement=args.entanglement,
        backend=args.backend,
        diff_method=args.diff_method,
        normalize_input=True,
        angle_scale=args.angle_scale,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_loader = _loader(z, spectra, labels, split["train"], args.batch_size, True)
    val_loader = _loader(z, spectra, labels, split["validation"], args.batch_size, False)
    test_loader = _loader(z, spectra, labels, split["test"], args.test_batch_size, False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    best_epoch = 0
    best_log: dict[str, Any] | None = None
    stale = 0
    logs: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        _set_residual_warmup(model, epoch, args.residual_warmup_epochs)
        train_stats = _train_epoch(model, train_loader, criterion, optimizer, device, args.metric_weight, args.temperature)
        val_loss, y_val, pred_val = _evaluate(model, val_loader, criterion, device)
        val_metrics = classification_metrics(y_val, pred_val, labels=list(range(num_classes)))
        log_row = {
            "dataset": dataset_name,
            "model": _model_name(args),
            "shot": shot,
            "seed": seed,
            "epoch": epoch,
            "residual_scale": float(model.residual_scale().detach().cpu().item()),
            "residual_warmup_factor": float(model.residual_warmup_factor.detach().cpu().item()),
            **train_stats,
            "val_loss": val_loss,
            "val_OA": val_metrics["OA"],
            "val_AA": val_metrics["AA"],
            "val_Kappa": val_metrics["Kappa"],
            "val_Macro-F1": val_metrics["Macro-F1"],
            "val_Weighted-F1": val_metrics["Weighted-F1"],
        }
        logs.append(log_row)
        monitor = val_metrics["OA"] if args.monitor == "oa" else val_metrics["Macro-F1"]
        if monitor > best_metric:
            best_metric = monitor
            best_epoch = epoch
            best_log = log_row
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    train_time = time.time() - started
    model.load_state_dict(best_state)
    model_name = _model_name(args)
    json_model_name = _json_model_name(args)
    stem = f"{dataset_name}_{model_name}_shot{shot}_seed{seed}"
    ckpt_path = out / "checkpoints" / f"{stem}.pt"
    torch.save(best_state, ckpt_path)
    test_started = time.time()
    test_loss, y_test, pred_test = _evaluate(model, test_loader, criterion, device)
    test_time = time.time() - test_started
    metrics = classification_metrics(y_test, pred_test, labels=list(range(num_classes)))
    residual_scale = float(model.residual_scale().detach().cpu().item())
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(num_classes)}
    per_class = per_class_metrics(y_test, pred_test, class_names)
    per_class["accuracy"] = per_class["recall"]
    cm = confusion_matrix(y_test, pred_test, labels=list(range(num_classes)))
    norm_cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    per_class.to_csv(out / "metrics" / f"{stem}_per_class.csv", index=False)
    per_class.to_csv(out / "metrics" / f"{stem}_per_class_metrics.csv", index=False)
    pd.DataFrame(logs).to_csv(out / "logs" / f"{stem}.csv", index=False)
    pd.DataFrame(cm).to_csv(out / "confusion_matrices" / f"{stem}.csv", index=False)
    pd.DataFrame(norm_cm).to_csv(out / "confusion_matrices" / f"{stem}_normalized.csv", index=False)
    _write_gate_values(out, stem, dataset_name, model_name, shot, seed, model, test_loader, device)

    setting_dir = out / "raw" / dataset_name / f"{shot}shot" / f"seed{seed}"
    setting_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(setting_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm).to_csv(setting_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(norm_cm).to_csv(setting_dir / "normalized_confusion_matrix.csv", index=False)
    pd.DataFrame(logs).to_csv(setting_dir / "training_log.csv", index=False)
    _write_training_config(setting_dir, args)

    payload = {
        "dataset": dataset_name,
        "model": json_model_name,
        "model_short": model_name,
        "qnn_variant": args.qnn_variant,
        "residual_scale_mode": args.residual_scale_mode,
        "residual_alpha_init": args.residual_alpha_init,
        "residual_warmup_epochs": args.residual_warmup_epochs,
        "residual_scale_final": residual_scale,
        "shot": shot,
        "seed": seed,
        "loss_mode": "supcon",
        "metric_weight": args.metric_weight,
        "temperature": args.temperature,
        "num_classes": num_classes,
        "pca_fit_scope": "full_image_unsupervised",
        "pca_evr_sum": pca_evr_sum,
        "patch_size": args.patch_size,
        "pca_bands": args.pca_bands,
        "best_epoch": best_epoch,
        "train_loss": None if best_log is None else best_log["train_loss"],
        "val_loss": None if best_log is None else best_log["val_loss"],
        "test_loss": test_loss,
        "train_time_seconds": train_time,
        "test_time_seconds": test_time,
        "trainable_parameters": param_count,
        "feature_path": str(feature_path),
        "checkpoint": str(ckpt_path),
        **metrics,
    }
    write_json(out / "metrics" / f"{stem}.json", payload)
    write_json(setting_dir / "metrics.json", payload)
    print(
        f"{dataset_name} {model_name} {shot}-shot seed{seed}: "
        f"OA={metrics['OA'] * 100:.2f} Macro-F1={metrics['Macro-F1'] * 100:.2f} best_epoch={best_epoch}"
    )
    return {
        "dataset": dataset_name,
        "model": model_name,
        "shot": shot,
        "seed": seed,
        **metrics,
        "best_epoch": best_epoch,
        "train_time_seconds": train_time,
        "test_time_seconds": test_time,
        "trainable_parameters": param_count,
        "train_size": len(split["train"]),
        "validation_size": len(split["validation"]),
        "test_size": len(split["test"]),
        "residual_scale_final": residual_scale,
    }


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric_weight: float,
    temperature: float,
) -> dict[str, float]:
    model.train()
    loss_sum = ce_sum = metric_sum = 0.0
    correct = count = 0
    for z, spectra, y, _ in loader:
        z = z.to(device)
        spectra = spectra.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(z, spectra)
        ce_loss = criterion(logits, y)
        features = model.fused_features(z, spectra)
        metric_loss = _supcon_loss(features, y, temperature)
        loss = ce_loss + float(metric_weight) * metric_loss
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(y)
        ce_sum += float(ce_loss.item()) * len(y)
        metric_sum += float(metric_loss.item()) * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        count += len(y)
    return {
        "train_loss": loss_sum / max(count, 1),
        "train_ce_loss": ce_sum / max(count, 1),
        "train_supcon_loss": metric_sum / max(count, 1),
        "train_accuracy": correct / max(count, 1),
    }


def _supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    batch = labels.shape[0]
    eye = torch.eye(batch, dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    logits_mask = ~eye
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return features.new_tensor(0.0)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1)[valid] / positive_count[valid].float()
    return -mean_log_prob_pos.mean()


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    loss_sum = 0.0
    count = 0
    ys, preds = [], []
    with torch.no_grad():
        for z, spectra, y, _ in loader:
            z = z.to(device)
            spectra = spectra.to(device)
            yy = y.to(device)
            logits = model(z, spectra)
            loss_sum += float(criterion(logits, yy).item()) * len(y)
            count += len(y)
            ys.append(y.numpy())
            preds.append(logits.argmax(1).cpu().numpy())
    return loss_sum / max(count, 1), np.concatenate(ys), np.concatenate(preds)


def _set_residual_warmup(model: nn.Module, epoch: int, warmup_epochs: int) -> None:
    if not hasattr(model, "set_residual_warmup_factor"):
        return
    if warmup_epochs <= 0:
        model.set_residual_warmup_factor(1.0)
        return
    model.set_residual_warmup_factor(min(1.0, float(epoch) / float(warmup_epochs)))


def _loader(z: np.ndarray, spectra: np.ndarray, labels: np.ndarray, indices: list[int], batch_size: int, shuffle: bool):
    return DataLoader(FeatureDataset(z, spectra, labels, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _copy_cached_features_if_available(args: argparse.Namespace, out: Path, dataset: str, shot: int, seed: int) -> None:
    target = out / "features" / f"{dataset}_shot{shot}_seed{seed}_features.npz"
    if target.exists() and not args.rebuild_features:
        return
    for root in args.feature_cache_dirs:
        path = Path(root) / "features" / f"{dataset}_shot{shot}_seed{seed}_features.npz"
        if path.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            return


def _write_outputs(out: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    pd.DataFrame(
        failures,
        columns=["dataset", "shot", "seed", "model", "stage", "status", "reason"],
    ).to_csv(out / "failed_runs.csv", index=False)
    write_json(out / "failed_runs.json", {"failed_runs": failures})
    if not rows:
        return
    all_runs = pd.DataFrame(rows).sort_values(["dataset", "shot", "seed"])
    all_runs.to_csv(out / "metrics" / "all_runs_metric_qnn.csv", index=False)
    all_runs.to_csv(out / "seedwise_results.csv", index=False)

    summary_rows = []
    for (dataset, shot, model), group in all_runs.groupby(["dataset", "shot", "model"]):
        row = {"dataset": dataset, "model": model, "shot": int(shot), "runs": len(group)}
        for metric in METRICS:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        row["mean_best_epoch"] = float(group["best_epoch"].mean())
        row["trainable_parameters"] = int(group["trainable_parameters"].iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "shot", "model"])
    summary.to_csv(out / "metrics" / "summary_by_dataset_shot_metric_qnn.csv", index=False)
    summary.to_csv(out / "summary_by_dataset_shot_metric_qnn.csv", index=False)
    _write_task_tables(out, all_runs, summary)
    _write_report(out, all_runs, summary, failures)


def _load_existing_rows(out: Path) -> list[dict[str, Any]]:
    path = out / "metrics" / "all_runs_metric_qnn.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _completed(out: Path, dataset: str, shot: int, seed: int, args: argparse.Namespace) -> bool:
    stem = f"{dataset}_{_model_name(args)}_shot{shot}_seed{seed}"
    return (
        (out / "metrics" / f"{stem}.json").exists()
        and (out / "metrics" / f"{stem}_per_class.csv").exists()
        and (out / "confusion_matrices" / f"{stem}.csv").exists()
        and (out / "confusion_matrices" / f"{stem}_normalized.csv").exists()
    )


def _write_task_tables(out: Path, all_runs: pd.DataFrame, summary: pd.DataFrame) -> None:
    rows = []
    for _, row in all_runs.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "shot": int(row["shot"]),
                "model": row["model"],
                "loss_type": "SupCon",
                "seed": int(row["seed"]),
                "oa": float(row["OA"]),
                "aa": float(row["AA"]),
                "kappa": float(row["Kappa"]),
                "macro_f1": float(row["Macro-F1"]),
                "weighted_f1": float(row["Weighted-F1"]),
            }
        )
    pd.DataFrame(rows).to_csv(out / "supcon_cross_dataset_summary.csv", index=False)

    agg_rows = []
    for _, row in summary.iterrows():
        agg_rows.append(
            {
                "dataset": row["dataset"],
                "shot": int(row["shot"]),
                "model": row["model"],
                "loss_type": "SupCon",
                "runs": int(row["runs"]),
                "oa_mean": row["mean_OA"],
                "oa_std": row["std_OA"],
                "aa_mean": row["mean_AA"],
                "aa_std": row["std_AA"],
                "kappa_mean": row["mean_Kappa"],
                "kappa_std": row["std_Kappa"],
                "macro_f1_mean": row["mean_Macro-F1"],
                "macro_f1_std": row["std_Macro-F1"],
                "weighted_f1_mean": row["mean_Weighted-F1"],
                "weighted_f1_std": row["std_Weighted-F1"],
            }
        )
    pd.DataFrame(agg_rows).to_csv(out / "supcon_cross_dataset_aggregate.csv", index=False)


def _write_report(out: Path, all_runs: pd.DataFrame, summary: pd.DataFrame, failures: list[dict[str, Any]]) -> None:
    display = summary.copy()
    for column in display.columns:
        if column.startswith("mean_") or column.startswith("std_"):
            if column != "mean_best_epoch":
                display[column] = (display[column] * 100).round(2)
            else:
                display[column] = display[column].round(2)
    lines = [
        "# Spectral QNN Gated Fusion + SupCon Cross-dataset Runs",
        "",
        "## Completed Runs",
        "",
        display.to_markdown(index=False) if not display.empty else "No completed runs.",
        "",
        "## Seedwise Results",
        "",
        all_runs.to_markdown(index=False) if len(all_runs) <= 30 else all_runs.head(30).to_markdown(index=False),
        "",
        "## Failures",
        "",
        pd.DataFrame(failures).to_markdown(index=False) if failures else "No failed runs.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_training_config(out: Path, args: argparse.Namespace) -> None:
    lines = []
    for key, value in vars(args).items():
        if isinstance(value, list):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {value}")
    (out / "training_config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _model_name(args: argparse.Namespace) -> str:
    residual = "_residualsafe" if args.residual_scale_mode == "learnable_sigmoid" else ""
    if args.qnn_variant == "standard":
        return f"spectral_qnn{residual}_gated_supcon"
    return f"spectral_qnn_{args.qnn_variant}{residual}_gated_supcon"


def _json_model_name(args: argparse.Namespace) -> str:
    residual = "_residualsafe" if args.residual_scale_mode == "learnable_sigmoid" else ""
    if args.qnn_variant == "standard":
        return f"hybridsn_small_spectral_qnn{residual}_gated_fusion_supcon"
    return f"hybridsn_small_spectral_qnn_{args.qnn_variant}{residual}_gated_fusion_supcon"


def _prepare_output(out: Path) -> None:
    for subdir in ("features", "checkpoints", "logs", "metrics", "confusion_matrices", "raw"):
        (out / subdir).mkdir(parents=True, exist_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled Pavia/Salinas Spectral QNN Gated Fusion + SupCon experiments.")
    parser.add_argument("--datasets", nargs="+", default=["pavia_university", "salinas"], choices=["indian_pines", "pavia_university", "salinas"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--shots", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--encoder_checkpoint_dir", default="result/hybridsn_small_fewshot_pavia_salinas_5_10shot/checkpoints")
    parser.add_argument("--split_dir", default="result/hybridsn_small_fewshot_pavia_salinas_5_10shot/split_indices")
    parser.add_argument("--output_dir", default="result/supcon_cross_dataset_pavia_salinas")
    parser.add_argument(
        "--feature_cache_dirs",
        nargs="*",
        default=[
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_salinas_5_10shot",
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_seed4_5shot",
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_10shot_seed0_2",
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_10shot_seed3",
            "result/hybridsn_small_spectral_qnn_gated_proto_pavia_10shot_seed4",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_partial",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_seed1_2_5shot",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_seed3_4_5shot",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_10shot_partial",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_10shot_seed1_2",
            "result/hybridsn_small_spectral_qnn_gated_proto_salinas_10shot_seed3_4",
        ],
    )
    parser.add_argument("--patch_size", type=int, default=19)
    parser.add_argument("--pca_bands", type=int, default=30)
    parser.add_argument("--conv3d_channels", nargs=3, type=int, default=[8, 16, 16])
    parser.add_argument("--conv2d_channels", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--metric_weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gate_mode", choices=["scalar", "classwise"], default="classwise")
    parser.add_argument("--qnn_variant", choices=["standard", "reupload_multiobs"], default="standard")
    parser.add_argument("--residual_scale_mode", choices=["none", "learnable_sigmoid"], default="none")
    parser.add_argument("--residual_alpha_init", type=float, default=-4.0)
    parser.add_argument("--residual_warmup_epochs", type=int, default=0)
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--qnn_layers", type=int, default=1)
    parser.add_argument("--entanglement", default="linear")
    parser.add_argument("--backend", default="lightning.qubit")
    parser.add_argument("--diff_method", default="adjoint")
    parser.add_argument("--angle_scale", type=float, default=float(np.pi))
    parser.add_argument("--monitor", choices=["macro_f1", "oa"], default="macro_f1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Unsupervised full-image PCA seed.")
    parser.add_argument("--rebuild_features", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip dataset/shot/seed runs with completed metric and confusion files.")
    return parser


if __name__ == "__main__":
    main()
