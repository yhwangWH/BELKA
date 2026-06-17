"""
评估模块
========

实现多蛋白质分层交叉验证评估:

  1. 逐折训练-预测-评估
  2. 每个蛋白质的 ROC-AUC
  3. 平均 ROC-AUC (比赛主要指标)
  4. OOF (Out-of-Fold) 预测收集
  5. 训练曲线可视化

比赛评估指标: mean per-protein ROC-AUC
  - 对每个蛋白质分别计算 ROC-AUC
  - 取三个 AUC 的算术平均作为最终得分
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import PROTEIN_NAMES, N_FOLDS


def evaluate_per_protein(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_labels: np.ndarray,
) -> Dict[str, float]:
    """
    计算每个蛋白质的 ROC-AUC 和平均 AUC.

    Parameters
    ----------
    y_true : np.ndarray
        真实标签 (0/1).
    y_pred : np.ndarray
        预测概率.
    protein_labels : np.ndarray
        每条样本对应的蛋白质名称.

    Returns
    -------
    metrics : dict
        包含各蛋白质 AUC 和 mean_auc 的字典.
    """
    metrics = {}
    aucs = []

    for protein in PROTEIN_NAMES:
        mask = protein_labels == protein
        if mask.sum() == 0:
            metrics[f"{protein}_auc"] = np.nan
            continue

        y_t = y_true[mask].astype(int)
        y_p = y_pred[mask]

        # 跳过只有一个类别的蛋白质 (无法计算 AUC)
        if len(np.unique(y_t)) < 2:
            print(f"  [{protein}] 只有一个类别, AUC 设为 NaN")
            metrics[f"{protein}_auc"] = np.nan
            continue

        auc = roc_auc_score(y_t, y_p)
        metrics[f"{protein}_auc"] = auc
        aucs.append(auc)

    # 平均 AUC (忽略 NaN)
    valid_aucs = [a for a in aucs if not np.isnan(a)]
    metrics["mean_auc"] = np.mean(valid_aucs) if valid_aucs else np.nan

    return metrics


def cross_validation_summary(
    fold_metrics: List[Dict[str, float]],
    label: str = "Validation",
) -> Dict[str, Dict[str, float]]:
    """
    汇总交叉验证结果, 计算均值与标准差.

    Parameters
    ----------
    fold_metrics : list of dict
        每折的评估指标字典列表.
    label : str
        标签名 (如 "Validation", "Train").

    Returns
    -------
    summary : dict
        每个指标包含 mean 和 std 的汇总.
    """
    if not fold_metrics:
        return {}

    all_keys = fold_metrics[0].keys()
    summary = {}

    for key in all_keys:
        values = [m[key] for m in fold_metrics if not np.isnan(m[key])]
        if values:
            summary[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        else:
            summary[key] = {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    return summary


def print_cv_summary(
    fold_metrics: List[Dict[str, float]],
    label: str = "Validation",
) -> None:
    """
    格式化打印交叉验证汇总结果.

    Parameters
    ----------
    fold_metrics : list of dict
        每折的评估指标字典.
    label : str
        标签名.
    """
    summary = cross_validation_summary(fold_metrics, label)

    print(f"\n{'='*60}")
    print(f"  {label} Cross-Validation Summary ({len(fold_metrics)} folds)")
    print(f"{'='*60}")

    for key, stats in summary.items():
        if "protein" in key.lower() or key == "mean_auc":
            print(f"  {key:15s}: {stats['mean']:.6f} ± {stats['std']:.6f} "
                  f"[{stats['min']:.6f}, {stats['max']:.6f}]")

    print(f"{'='*60}\n")


def plot_training_curves(
    evals_result: Dict[str, Dict[str, list]],
    fold: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    绘制 LightGBM 训练曲线 (AUC vs iterations).

    Parameters
    ----------
    evals_result : dict
        LightGBM 训练返回的 evals_result_.
    fold : int, optional
        折号 (用于标题).
    save_path : str, optional
        图片保存路径.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for ds_name, metrics in evals_result.items():
        if "auc" in metrics:
            ax.plot(metrics["auc"], label=f"{ds_name}")

    title = "Training Curves"
    if fold is not None:
        title += f" (Fold {fold})"
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"[Eval] 训练曲线已保存至: {save_path}")
    else:
        plt.show()

    plt.close()


def compute_additional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算额外评估指标: Precision, Recall, F1, PR-AUC.

    Parameters
    ----------
    y_true : np.ndarray
        真实标签.
    y_pred : np.ndarray
        预测概率.
    threshold : float
        二分类阈值.

    Returns
    -------
    metrics : dict
        评估指标字典.
    """
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
    )

    y_pred_class = (y_pred >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred_class, zero_division=0),
        "recall": recall_score(y_true, y_pred_class, zero_division=0),
        "f1": f1_score(y_true, y_pred_class, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_pred),
    }

    return metrics
