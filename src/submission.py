"""
提交文件生成模块
================

负责:
  1. 收集所有折的 OOF 预测 (用于验证分数校验)
  2. 生成符合 Kaggle 格式的提交文件
  3. 预测值裁剪 (clip 到 [1e-6, 1-1e-6])
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from .config import SUBMISSION_DIR, PROTEIN_NAMES


def clip_predictions(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    将预测概率裁剪到安全范围, 避免 log loss/auc 计算出错.

    Parameters
    ----------
    probs : np.ndarray
        原始预测概率.
    eps : float
        裁剪下限 (预测值会被限制在 [eps, 1-eps]).

    Returns
    -------
    clipped : np.ndarray
        裁剪后的预测.
    """
    return np.clip(probs, eps, 1.0 - eps)


def create_submission(
    test_ids: np.ndarray,
    test_preds: np.ndarray,
    sample_sub_path: str,
    output_name: str = "submission_phase1_baseline.csv",
) -> str:
    """
    生成 Kaggle 提交文件.

    Parameters
    ----------
    test_ids : np.ndarray
        测试样本的 id 列.
    test_preds : np.ndarray
        模型预测概率.
    sample_sub_path : str
        Kaggle 官方提交模板路径 (用于确保格式一致).
    output_name : str
        输出文件名.

    Returns
    -------
    output_path : str
        生成的提交文件完整路径.
    """
    # 裁剪预测值
    test_preds = clip_predictions(test_preds)

    # 读取官方模板确保 id 顺序完全一致
    sample_sub = pd.read_csv(sample_sub_path)
    expected_n = len(sample_sub)

    if len(test_preds) != expected_n:
        raise ValueError(
            f"预测数量 ({len(test_preds)}) 与模板行数 ({expected_n}) 不匹配!"
        )

    # 创建提交文件
    submission = sample_sub.copy()
    submission["binds"] = test_preds

    # 保存
    output_path = os.path.join(SUBMISSION_DIR, output_name)
    submission.to_csv(output_path, index=False)
    print(f"[Submission] 提交文件已生成: {output_path}")
    print(f"  行数: {len(submission):,}")
    print(f"  预测范围: [{test_preds.min():.6f}, {test_preds.max():.6f}]")
    print(f"  预测均值: {test_preds.mean():.6f}")

    return output_path


def collect_oof_predictions(
    fold_predictions: List[Dict],
    df_oof: pd.DataFrame,
) -> pd.DataFrame:
    """
    收集所有折的 OOF (Out-of-Fold) 预测到 DataFrame 中.

    每一折对验证集产生预测, 汇总后得到整个训练集的 OOF 预测,
    可用于整体评估和模型融合的 meta-feature.

    Parameters
    ----------
    fold_predictions : list of dict
        每折的预测结果, 格式:
        [{"valid_idx": ..., "y_valid": ..., "pred_prob": ..., "protein": ...}, ...]
    df_oof : pd.DataFrame
        原始数据 (用于获取蛋白质列).

    Returns
    -------
    oof_df : pd.DataFrame
        增加 "oof_pred" 列的数据框.
    """
    oof_df = df_oof.copy()
    oof_df["oof_pred"] = np.nan

    for fp in fold_predictions:
        valid_idx = fp["valid_idx"]
        # 确保索引对齐
        if isinstance(valid_idx, np.ndarray):
            oof_df.iloc[valid_idx, oof_df.columns.get_loc("oof_pred")] = fp["pred_prob"]
        else:
            oof_df.loc[valid_idx, "oof_pred"] = fp["pred_prob"]

    # 裁剪 OOF 预测
    oof_df["oof_pred"] = clip_predictions(oof_df["oof_pred"].values)

    n_missing = oof_df["oof_pred"].isna().sum()
    if n_missing > 0:
        print(f"[Submission]  警告: {n_missing} 条 OOF 预测缺失")

    return oof_df


def compute_oof_score(
    oof_df: pd.DataFrame,
    label_col: str = "binds",
    protein_col: str = "protein_name",
    pred_col: str = "oof_pred",
) -> Dict[str, float]:
    """
    使用 OOF 预测计算整体的 per-protein AUC.

    Parameters
    ----------
    oof_df : pd.DataFrame
        包含标签和 OOF 预测的 DataFrame.
    label_col : str
        标签列名.
    protein_col : str
        蛋白质列名.
    pred_col : str
        OOF 预测列名.

    Returns
    -------
    scores : dict
        各蛋白质 AUC 和 mean_auc.
    """
    from .evaluation import evaluate_per_protein

    # 移除缺失值
    valid = oof_df.dropna(subset=[pred_col])
    print(f"[Submission] OOF 评估: {len(valid):,}/{len(oof_df):,} 条有效预测")

    scores = evaluate_per_protein(
        y_true=valid[label_col].values,
        y_pred=valid[pred_col].values,
        protein_labels=valid[protein_col].values,
    )

    print(f"\n[Submission] OOF Score:")
    for protein in PROTEIN_NAMES:
        key = f"{protein}_auc"
        print(f"  {protein:6s} AUC: {scores.get(key, np.nan):.6f}")
    print(f"  Mean  AUC: {scores.get('mean_auc', np.nan):.6f}")

    return scores


def blend_predictions(
    preds_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    模型融合: 加权平均多个模型的预测.

    Parameters
    ----------
    preds_list : list of np.ndarray
        多个模型的预测概率列表.
    weights : list of float, optional
        每个模型的权重. None 表示等权平均.

    Returns
    -------
    blended : np.ndarray
        融合后的预测.
    """
    if weights is None:
        weights = [1.0] * len(preds_list)

    # 归一化权重
    weights = np.array(weights) / np.sum(weights)

    blended = np.zeros_like(preds_list[0])
    for pred, w in zip(preds_list, weights):
        blended += w * pred

    return blended
