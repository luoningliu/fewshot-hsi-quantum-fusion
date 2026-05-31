from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.spatial_split_hybridsn_indian_pines import _make_spatial_split
from src.datasets.hsi_dataset import load_hsi_mat
from src.utils.config import load_yaml


OUT = Path("result/spatial_isolation_summary_20260531")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml("configs/experiments/spatial_split_hybridsn_indian_pines.yaml")
    data_cfg = load_yaml(cfg["dataset"]["config"])
    raw = load_hsi_mat(data_cfg)
    rows, cols = np.nonzero(raw.gt != raw.background_label)
    labels = raw.gt[rows, cols].astype(np.int64) - 1
    class_names = {i: data_cfg["class_names"][i + 1] for i in range(int(data_cfg["num_classes"]))}

    coverage = []
    for seed in cfg["spatial_split"]["seeds"]:
        split = _make_spatial_split(
            rows,
            cols,
            labels,
            raw.gt.shape,
            int(cfg["spatial_split"]["grid_rows"]),
            int(cfg["spatial_split"]["grid_cols"]),
            seed=int(seed),
            train_block_fraction=float(cfg["spatial_split"]["train_block_fraction"]),
            validation_block_fraction=float(cfg["spatial_split"]["validation_block_fraction"]),
        )
        for split_name, indices in split.items():
            counts = np.bincount(labels[np.asarray(indices)], minlength=int(data_cfg["num_classes"]))
            for class_id, count in enumerate(counts):
                coverage.append(
                    {
                        "seed": int(seed),
                        "split": split_name,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "count": int(count),
                    }
                )
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(OUT / "spatial_split_class_coverage.csv", index=False)

    hybrid_runs = pd.read_csv("result/spatial_split_hybridsn_indian_pines/all_runs.csv")
    hybrid_best = json.loads(Path("result/spatial_split_hybridsn_indian_pines/best_metrics.json").read_text())
    qnn_heads = pd.read_csv("result/spatial_split_qnn_heads_indian_pines/all_runs.csv")
    qnn_best = json.loads(Path("result/spatial_split_qnn_heads_indian_pines/best_metrics.json").read_text())

    random_reference = pd.DataFrame(
        [
            {
                "protocol": "random pixel split",
                "model": "Tuned HybridSN",
                "OA": 0.9880,
                "AA": 0.9727,
                "Macro-F1": 0.9719,
                "Weighted-F1": 0.9881,
                "note": "from existing random pixel split final report",
            },
            {
                "protocol": "few-shot random pixel split",
                "model": "HybridSN-small 10-shot",
                "OA": 0.8012,
                "AA": 0.8864,
                "Macro-F1": 0.7153,
                "Weighted-F1": 0.8087,
                "note": "mean over seeds 0-4",
            },
            {
                "protocol": "few-shot random pixel split",
                "model": "Spectral QNN + SupCon 10-shot",
                "OA": 0.8107,
                "AA": 0.8905,
                "Macro-F1": 0.7226,
                "Weighted-F1": 0.8179,
                "note": "mean over seeds 0-4",
            },
            {
                "protocol": "spatial block split",
                "model": "HybridSN spatial split",
                "OA": hybrid_best["OA"],
                "AA": hybrid_best["AA"],
                "Macro-F1": hybrid_best["Macro-F1"],
                "Weighted-F1": hybrid_best["Weighted-F1"],
                "note": "strict grid-block split, seed selected by validation Macro-F1",
            },
            {
                "protocol": "spatial block split",
                "model": qnn_best["model"],
                "OA": qnn_best["OA"],
                "AA": qnn_best["AA"],
                "Macro-F1": qnn_best["Macro-F1"],
                "Weighted-F1": qnn_best["Weighted-F1"],
                "note": "frozen spatial encoder with head comparison; best head is validation-selected",
            },
        ]
    )
    random_reference.to_csv(OUT / "spatial_vs_random_summary.csv", index=False)

    write_report(coverage_df, hybrid_runs, hybrid_best, qnn_heads, qnn_best, random_reference)


def write_report(
    coverage_df: pd.DataFrame,
    hybrid_runs: pd.DataFrame,
    hybrid_best: dict,
    qnn_heads: pd.DataFrame,
    qnn_best: dict,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# 空间隔离实验汇总",
        "",
        "## 实验目的",
        "",
        "该实验用于回应 random pixel split 可能存在空间邻域泄漏的问题。空间隔离实验将 Indian Pines 按 5 x 5 网格划分为空间块，训练、验证、测试使用互不重叠的 block，因此测试样本与训练样本在空间上隔离。",
        "",
        "## 划分设置",
        "",
        "- 数据集：Indian Pines。",
        "- Grid：5 x 5 spatial blocks。",
        "- 训练 block 比例：0.20。",
        "- 验证 block 比例：0.12。",
        "- 测试 block：剩余 block。",
        "- HybridSN spatial split seeds：0, 1。",
        "- QNN head pilot：seed 0，先在 spatial train blocks 上训练 encoder，再冻结 encoder 比较 linear / MLP / residual QNN / gated residual QNN heads。",
        "",
        "重要限制：该 strict block split 不强制每个类别都出现在训练区域，因此是较严格的空间泛化诊断，不应与 random pixel split 直接作为同一难度协议比较。",
        "",
        "## 空间隔离 HybridSN 结果",
        "",
        markdown_table(percent_table(hybrid_runs, ["best_val_macro_f1", "best_val_oa", "best_val_aa"])),
        "",
        f"Best spatial test: OA={hybrid_best['OA']*100:.2f}, AA={hybrid_best['AA']*100:.2f}, Macro-F1={hybrid_best['Macro-F1']*100:.2f}, Weighted-F1={hybrid_best['Weighted-F1']*100:.2f}.",
        "",
        "## 空间隔离 QNN head pilot",
        "",
        markdown_table(percent_table(qnn_heads, ["best_val_macro_f1", "best_val_oa", "best_val_aa"])),
        "",
        f"Best head test: {qnn_best['model']}, OA={qnn_best['OA']*100:.2f}, AA={qnn_best['AA']*100:.2f}, Macro-F1={qnn_best['Macro-F1']*100:.2f}, Weighted-F1={qnn_best['Weighted-F1']*100:.2f}.",
        "",
        "QNN heads did not outperform the MLP head under this spatial split pilot. The best validation head is `mlp_h64`; residual QNN and gated residual QNN have lower validation Macro-F1 and much higher training cost.",
        "",
        "## Random Split 与 Spatial Split 对照",
        "",
        markdown_table(percent_table(comparison, ["OA", "AA", "Macro-F1", "Weighted-F1"])),
        "",
        "## Class Coverage 诊断",
        "",
    ]
    coverage_summary = (
        coverage_df.pivot_table(index=["seed", "class_name"], columns="split", values="count", fill_value=0)
        .reset_index()
        .sort_values(["seed", "class_name"])
    )
    missing_train = coverage_summary[coverage_summary["train"] == 0]
    lines.extend(
        [
            f"- seed 0/1 中，训练 split 缺失类别总记录数：{len(missing_train)}。",
            "- 这解释了空间隔离下 Macro-F1 和 AA 大幅下降：部分测试类别在训练区域中没有样本，模型无法学习这些类。",
            "",
            "训练集中缺失的类别：",
            "",
            markdown_table(missing_train[["seed", "class_name", "train", "validation", "test"]]),
            "",
            "## 结论",
            "",
            "1. random pixel split 的结果显著高于 spatial block split，说明原随机像素协议可能高估了 patch-based HSI 模型的真实空间泛化能力。",
            "2. 在 Indian Pines strict spatial split 下，HybridSN spatial test OA 仅 33.16%，Macro-F1 仅 13.03%，远低于 random pixel split 的 tuned HybridSN OA 98.80%、Macro-F1 97.19%。",
            "3. QNN head pilot 也未显示空间隔离优势；最佳 head 是 MLP，QNN residual head 的验证 Macro-F1 低于 MLP。",
            "4. 当前 spatial split 是严格诊断，不是最终主实验协议。正式论文中应把它作为补充实验，用来说明 random pixel split 的局限和模型空间泛化难度，而不是用它否定 few-shot 主结果。",
            "5. 后续若要做更公平的 spatial few-shot 主实验，需要设计 class-balanced spatial split，保证每个类别在 train/validation/test 中均有样本。",
            "",
        ]
    )
    (OUT / "spatial_isolation_report.md").write_text("\n".join(lines), encoding="utf-8")


def percent_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col] * 100
    return out


def markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: f"{x:.2f}")
    header = "| " + " | ".join(show.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in show.to_numpy()]
    return "\n".join([header, sep, *rows])


if __name__ == "__main__":
    main()
