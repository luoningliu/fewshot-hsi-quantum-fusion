from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.analysis.metrics import classification_metrics, per_class_metrics, write_json
from src.models.classical import BottleneckClassifier, CNN1D, CNN2D, CNN3D, HybridSN, HybridSNEncoder, MLPClassifier
from src.models.quantum import QNNClassifier
from src.utils.config import load_yaml
from src.utils.seed import set_seed


class HSINumpyDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, indices: list[int]):
        self.data = data
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[item]
        x = torch.from_numpy(self.data[idx]).float()
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def run(config_path: str | Path) -> None:
    config = load_yaml(config_path)
    if config["stage"] == "stage3_deep_baselines":
        run_stage3(config)
        return
    if config["stage"] == "stage4_qnn_classifier":
        run_stage4(config)
        return
    raise NotImplementedError(
        f"{config['stage']} is scaffolded but not implemented in the shared training runner yet."
    )


def run_stage3(config: dict[str, Any]) -> None:
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    training_config = config["training"]
    set_seed(int(training_config["seed"]))
    device = _resolve_device(str(training_config.get("device", "auto")))

    summary_rows = []
    for dataset_config_path in config["datasets"]:
        dataset_config = load_yaml(dataset_config_path)
        data = np.load(dataset_config["output"]["processed_npz"], allow_pickle=True)
        split = _load_json(dataset_config["output"]["split_json"])
        labels = data["y"].astype(np.int64)
        num_classes = int(dataset_config["num_classes"])

        for model_name in config["models"]:
            started = time.time()
            result_dir = output_root / dataset_config["dataset_id"] / model_name
            result_dir.mkdir(parents=True, exist_ok=True)
            model, inputs = _build_model(model_name, dataset_config, data)
            model.to(device)
            train_log, best_state = _fit_model(
                model=model,
                inputs=inputs,
                labels=labels,
                split=split,
                num_classes=num_classes,
                training_config=training_config,
                device=device,
            )
            model.load_state_dict(best_state)
            y_test, pred = _predict(
                model=model,
                inputs=inputs,
                labels=labels,
                indices=split["test"],
                batch_size=int(training_config["batch_size"]),
                device=device,
            )
            metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
            elapsed = time.time() - started
            row = {
                "dataset_id": dataset_config["dataset_id"],
                "model": model_name,
                **metrics,
                "training_time_seconds": elapsed,
                "device": str(device),
            }
            summary_rows.append(row)

            pd.DataFrame(train_log).to_csv(result_dir / "training_log.csv", index=False)
            write_json(result_dir / "metrics.json", row)
            write_json(
                result_dir / "runtime_summary.json",
                {"training_time_seconds": elapsed, "device": str(device)},
            )
            _write_model_summary(result_dir / "model_summary.txt", model)
            per_class_metrics(
                y_test,
                pred,
                {i: dataset_config["class_names"][i + 1] for i in range(num_classes)},
            ).to_csv(result_dir / "per_class_metrics.csv", index=False)
            cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
            pd.DataFrame(cm).to_csv(result_dir / "confusion_matrix.csv", index=False)
            norm_cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            pd.DataFrame(norm_cm).to_csv(result_dir / "normalized_confusion_matrix.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "summary_table.csv", index=False)
    _write_stage3_report(output_root / "report.md", summary)


def run_stage4(config: dict[str, Any]) -> None:
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    training_config = config["training"]
    set_seed(int(training_config["seed"]))
    device = _resolve_device(str(training_config.get("device", "auto")))
    qnn_datasets = set(training_config.get("qnn_datasets", []))

    summary_rows = []
    for dataset_config_path in config["datasets"]:
        dataset_config = load_yaml(dataset_config_path)
        dataset_id = dataset_config["dataset_id"]
        data = np.load(dataset_config["output"]["processed_npz"], allow_pickle=True)
        split = _load_json(dataset_config["output"]["split_json"])
        patches = data["x"]
        labels = data["y"].astype(np.int64)
        num_classes = int(dataset_config["num_classes"])
        pca_channels = int(dataset_config["pca_components"])

        encoder_dir = output_root / dataset_id / "hybridsn_linear"
        encoder_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = output_root / dataset_id / "hybridsn_embeddings.npz"
        reuse_existing = bool(training_config.get("reuse_existing", False))
        linear_metrics_path = encoder_dir / "metrics.json"
        if reuse_existing and embeddings_path.exists() and linear_metrics_path.exists():
            embeddings = np.load(embeddings_path)["z"].astype(np.float32)
            summary_rows.append(_load_json(linear_metrics_path))
        else:
            encoder_start = time.time()
            encoder_model = HybridSN(pca_channels=pca_channels, num_classes=num_classes).to(device)
            encoder_log, encoder_state = _fit_model(
                model=encoder_model,
                inputs=patches,
                labels=labels,
                split=split,
                num_classes=num_classes,
                training_config={
                    **training_config,
                    "epochs": int(training_config["encoder_epochs"]),
                    "batch_size": int(training_config["batch_size"]),
                    "learning_rate": float(training_config["learning_rate"]),
                },
                device=device,
            )
            encoder_model.load_state_dict(encoder_state)
            y_test, linear_pred = _predict(
                model=encoder_model,
                inputs=patches,
                labels=labels,
                indices=split["test"],
                batch_size=int(training_config["batch_size"]),
                device=device,
            )
            linear_metrics = classification_metrics(y_test, linear_pred, labels=list(range(num_classes)))
            linear_row = {
                "dataset_id": dataset_id,
                "model": "hybridsn_linear",
                **linear_metrics,
                "training_time_seconds": time.time() - encoder_start,
                "device": str(device),
            }
            summary_rows.append(linear_row)
            _write_eval_outputs(encoder_dir, linear_row, encoder_log, encoder_model, y_test, linear_pred, dataset_config)
            torch.save(encoder_model.state_dict(), encoder_dir / "model_state.pt")

            encoder = copy.deepcopy(encoder_model.encoder).to(device)
            embeddings = _extract_embeddings(
                encoder=encoder,
                patches=patches,
                labels=labels,
                indices=list(range(len(labels))),
                batch_size=int(training_config["batch_size"]),
                device=device,
            )
            np.savez_compressed(embeddings_path, z=embeddings, y=labels)

        head_specs: list[tuple[str, nn.Module, int, int, float]] = [
            (
                "hybridsn_mlp",
                MLPClassifier(input_dim=embeddings.shape[1], num_classes=num_classes),
                int(training_config["head_epochs"]),
                int(training_config["batch_size"]),
                float(training_config["learning_rate"]),
            ),
            (
                "hybridsn_bottleneck",
                BottleneckClassifier(input_dim=embeddings.shape[1], num_classes=num_classes, bottleneck_dim=int(config["qnn"]["qubits"])),
                int(training_config["head_epochs"]),
                int(training_config["batch_size"]),
                float(training_config["learning_rate"]),
            ),
        ]
        if dataset_id in qnn_datasets:
            head_specs.append(
                (
                    "hybridsn_qnn",
                    QNNClassifier(
                        input_dim=embeddings.shape[1],
                        num_classes=num_classes,
                        qubits=int(config["qnn"]["qubits"]),
                        layers=int(config["qnn"]["layers"]),
                        entanglement=str(config["qnn"]["entanglement"]),
                        backend=str(config["qnn"].get("backend", "lightning.qubit")),
                        diff_method=str(config["qnn"].get("diff_method", "adjoint")),
                    ),
                    int(training_config["qnn_epochs"]),
                    int(training_config["qnn_batch_size"]),
                    float(training_config["qnn_learning_rate"]),
                )
            )

        for model_name, head, epochs, batch_size, learning_rate in head_specs:
            result_dir = output_root / dataset_id / model_name
            metrics_path = result_dir / "metrics.json"
            if reuse_existing and model_name != "hybridsn_qnn" and metrics_path.exists():
                summary_rows.append(_load_json(metrics_path))
                continue
            started = time.time()
            result_dir.mkdir(parents=True, exist_ok=True)
            head = head.to(device)
            head_split = split
            note = None
            if model_name == "hybridsn_qnn" and "qnn_subset" in training_config:
                head_split = _make_stratified_subset_split(labels, split, training_config["qnn_subset"], seed=int(training_config["seed"]))
                note = "qnn_stratified_subset"
            head_log, head_state = _fit_model(
                model=head,
                inputs=embeddings,
                labels=labels,
                split=head_split,
                num_classes=num_classes,
                training_config={
                    **training_config,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                },
                device=device,
            )
            head.load_state_dict(head_state)
            y_test, pred = _predict(
                model=head,
                inputs=embeddings,
                labels=labels,
                indices=head_split["test"],
                batch_size=batch_size,
                device=device,
            )
            metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
            row = {
                "dataset_id": dataset_id,
                "model": model_name,
                **metrics,
                "training_time_seconds": time.time() - started,
                "device": str(device),
            }
            if note:
                row["note"] = note
                row["subset_train"] = len(head_split["train"])
                row["subset_validation"] = len(head_split["validation"])
                row["subset_test"] = len(head_split["test"])
            summary_rows.append(row)
            _write_eval_outputs(result_dir, row, head_log, head, y_test, pred, dataset_config)
            torch.save(head.state_dict(), result_dir / "model_state.pt")

        if dataset_id not in qnn_datasets:
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "model": "hybridsn_qnn",
                    "OA": np.nan,
                    "AA": np.nan,
                    "Kappa": np.nan,
                    "Macro-F1": np.nan,
                    "Weighted-F1": np.nan,
                    "training_time_seconds": np.nan,
                    "device": str(device),
                    "note": "skipped_in_cpu_pilot",
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "summary_table.csv", index=False)
    _write_stage_report(output_root / "report.md", "Stage 4 QNN Classifier Report", summary)


def _build_model(model_name: str, dataset_config: dict[str, Any], data: Any) -> tuple[nn.Module, np.ndarray]:
    pca_channels = int(dataset_config["pca_components"])
    num_classes = int(dataset_config["num_classes"])
    if model_name == "cnn1d":
        return CNN1D(input_channels=pca_channels, num_classes=num_classes), data["spectral"]
    if model_name == "cnn2d":
        return CNN2D(input_channels=pca_channels, num_classes=num_classes), data["x"]
    if model_name == "cnn3d":
        return CNN3D(input_channels=pca_channels, num_classes=num_classes), data["x"]
    if model_name == "hybridsn":
        return HybridSN(pca_channels=pca_channels, num_classes=num_classes), data["x"]
    raise ValueError(f"Unsupported Stage 3 model: {model_name}")


def _fit_model(
    model: nn.Module,
    inputs: np.ndarray,
    labels: np.ndarray,
    split: dict[str, Any],
    num_classes: int,
    training_config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
    train_loader = _make_loader(inputs, labels, split["train"], training_config, shuffle=True)
    val_loader = _make_loader(inputs, labels, split["validation"], training_config, shuffle=False)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    selection_metric = str(training_config.get("selection_metric", "validation_OA"))
    best_metric = -float("inf")
    stale_epochs = 0
    log_rows = []

    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(y)
            train_count += len(y)

        y_val, pred_val = _predict_loader(model, val_loader, device)
        val_metrics = classification_metrics(y_val, pred_val, labels=list(range(num_classes)))
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_count, 1),
            "validation_OA": val_metrics["OA"],
            "validation_AA": val_metrics["AA"],
            "validation_Kappa": val_metrics["Kappa"],
            "validation_Macro-F1": val_metrics["Macro-F1"],
        }
        log_rows.append(row)
        current_metric = float(row[selection_metric])
        if current_metric > best_metric:
            best_metric = current_metric
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(training_config["patience"]):
            break
    return log_rows, best_state


def _predict(
    model: nn.Module,
    inputs: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(HSINumpyDataset(inputs, labels, indices), batch_size=batch_size, shuffle=False)
    return _predict_loader(model, loader, device)


def _predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y_true.append(y.numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _extract_embeddings(
    encoder: HybridSNEncoder,
    patches: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(HSINumpyDataset(patches, labels, indices), batch_size=batch_size, shuffle=False)
    encoder.eval()
    chunks = []
    with torch.no_grad():
        for x, _ in loader:
            chunks.append(encoder(x.to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _make_loader(
    inputs: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    training_config: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        HSINumpyDataset(inputs, labels, indices),
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
    )


def _make_stratified_subset_split(
    labels: np.ndarray,
    split: dict[str, Any],
    subset_config: dict[str, int],
    seed: int,
) -> dict[str, list[int]]:
    return {
        "train": _sample_per_class(labels, split["train"], int(subset_config["train_per_class"]), seed + 11),
        "validation": _sample_per_class(labels, split["validation"], int(subset_config["validation_per_class"]), seed + 17),
        "test": _sample_per_class(labels, split["test"], int(subset_config["test_per_class"]), seed + 23),
    }


def _sample_per_class(labels: np.ndarray, indices: list[int], per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    indices_array = np.asarray(indices, dtype=np.int64)
    selected = []
    for label in sorted(np.unique(labels)):
        class_indices = indices_array[labels[indices_array] == label]
        if len(class_indices) == 0:
            continue
        sample_size = min(per_class, len(class_indices))
        selected.extend(rng.choice(class_indices, size=sample_size, replace=False).astype(int).tolist())
    rng.shuffle(selected)
    return sorted(selected)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_model_summary(path: Path, model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    path.write_text(f"{model}\n\nTotal parameters: {total}\nTrainable parameters: {trainable}\n", encoding="utf-8")


def _write_stage3_report(path: Path, summary: pd.DataFrame) -> None:
    _write_stage_report(path, "Stage 3 Deep Baselines Report", summary)


def _write_stage_report(path: Path, title: str, summary: pd.DataFrame) -> None:
    display = summary.copy()
    for col in ["OA", "AA", "Kappa", "Macro-F1", "Weighted-F1"]:
        if col in display:
            display[col] = (display[col] * 100).round(2)
    path.write_text(f"# {title}\n\n" + display.to_markdown(index=False) + "\n", encoding="utf-8")


def _write_eval_outputs(
    result_dir: Path,
    metrics_row: dict[str, Any],
    train_log: list[dict[str, float]],
    model: nn.Module,
    y_test: np.ndarray,
    pred: np.ndarray,
    dataset_config: dict[str, Any],
) -> None:
    num_classes = int(dataset_config["num_classes"])
    pd.DataFrame(train_log).to_csv(result_dir / "training_log.csv", index=False)
    write_json(result_dir / "metrics.json", metrics_row)
    write_json(
        result_dir / "runtime_summary.json",
        {
            "training_time_seconds": metrics_row.get("training_time_seconds"),
            "device": metrics_row.get("device"),
        },
    )
    _write_model_summary(result_dir / "model_summary.txt", model)
    per_class_metrics(
        y_test,
        pred,
        {i: dataset_config["class_names"][i + 1] for i in range(num_classes)},
    ).to_csv(result_dir / "per_class_metrics.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(result_dir / "confusion_matrix.csv", index=False)
    norm_cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(norm_cm).to_csv(result_dir / "normalized_confusion_matrix.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training stages.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
