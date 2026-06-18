"""
采样策略模块
=============

针对极度不平衡数据 (正样本 ~0.5%) 的分层采样策略:

  策略1: 正负比采样 (默认)
    选取全部正样本, 从负样本中随机采样, 控制正负比例 (如 1:20).

  策略2: 化学多样性采样 (可选, 二期启用)
    按 building block 分组, 确保各化学空间都有覆盖.

主要函数:
  - stratified_sample(): 按蛋白质分层 + 正负比采样的核心函数
  - create_folds(): 为交叉验证创建分层 KFold 划分
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List

from sklearn.model_selection import StratifiedKFold, GroupKFold

from .config import (
    PROTEIN_NAMES,
    NEGATIVE_RATIO,
    MAX_POSITIVE_SAMPLES,
    RANDOM_SEED,
    N_FOLDS,
)


def stratified_sample(
    df: pd.DataFrame,
    negative_ratio: int = NEGATIVE_RATIO,
    max_positive_samples: Optional[int] = MAX_POSITIVE_SAMPLES,
    random_state: int = RANDOM_SEED,
    protein_col: str = "protein_name",
    label_col: str = "binds",
) -> pd.DataFrame:
    """
    分层采样: 对每个蛋白质分别采样, 保持正负比一致.

    采样逻辑:
      1. 先取出所有正样本 (binds=1)
      2. 对每个蛋白质, 从对应负样本中随机采样 negative_ratio 倍
      3. 合并正负样本后打乱

    这样可以确保:
      - 所有正样本信息不丢失
      - 负样本数量可控, 训练速度合理
      - 每个蛋白质的正负比例一致

    Parameters
    ----------
    df : pd.DataFrame
        完整训练数据.
    negative_ratio : int
        负样本对正样本的倍数 (如 20 表示 1:20 的正负比).
    max_positive_samples : int, optional
        每个蛋白质最大正样本数. None 表示全部使用.
        在正样本很多时限制上限.
    random_state : int
        随机种子.

    Returns
    -------
    sampled_df : pd.DataFrame
        采样后的数据 (正样本 + 采样的负样本).
    """
    print(f"\n[Sampler] 分层采样: 正负比 1:{negative_ratio}")

    # 1. 分离正负样本 (布尔索引已返回副本, 无需 .copy())
    pos_samples = df[df[label_col] == 1]
    neg_samples = df[df[label_col] == 0]

    print(f"  原始数据: {len(df):,} 行")
    print(f"    正样本: {len(pos_samples):,} ({100*len(pos_samples)/len(df):.4f}%)")
    print(f"    负样本: {len(neg_samples):,}")

    # 2. 按蛋白质分组采样
    rng = np.random.RandomState(random_state)
    sampled_chunks: List[pd.DataFrame] = []

    for protein in PROTEIN_NAMES:
        # 该蛋白质的正样本
        prot_pos = pos_samples[pos_samples[protein_col] == protein]
        n_prot_pos = len(prot_pos)

        # 该蛋白质的负样本
        prot_neg = neg_samples[neg_samples[protein_col] == protein]
        n_prot_neg = len(prot_neg)

        # 限制正样本数量 (如配置了上限)
        if max_positive_samples is not None and n_prot_pos > max_positive_samples:
            prot_pos = prot_pos.sample(
                n=max_positive_samples, random_state=random_state
            )
            n_prot_pos = max_positive_samples

        # 计算需要的负样本数
        n_neg_needed = min(n_prot_neg, n_prot_pos * negative_ratio)

        # 从负样本中随机抽取
        if n_neg_needed < n_prot_neg:
            prot_neg = prot_neg.sample(n=n_neg_needed, random_state=random_state)

        print(f"    {protein:6s}: 正样本 {n_prot_pos:,} + 负样本 {len(prot_neg):,} "
              f"(1:{len(prot_neg) / max(n_prot_pos, 1):.1f})")

        sampled_chunks.append(prot_pos)
        sampled_chunks.append(prot_neg)

    # 3. 合并并打乱
    sampled_df = pd.concat(sampled_chunks, axis=0, ignore_index=True)
    # 释放中间采样缓冲, 避免 concat 期间峰值内存翻倍
    del sampled_chunks, pos_samples, neg_samples
    import gc
    gc.collect()
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    total_pos = (sampled_df[label_col] == 1).sum()
    total_neg = (sampled_df[label_col] == 0).sum()

    print(f"  采样后: {len(sampled_df):,} 行")
    print(f"    正样本: {total_pos:,} ({100 * total_pos / len(sampled_df):.2f}%)")
    print(f"    负样本: {total_neg:,} ({100 * total_neg / len(sampled_df):.2f}%)")
    print(f"    正负比: 1:{total_neg / max(total_pos, 1):.1f}")

    return sampled_df


def create_protein_stratified_folds(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_SEED,
    protein_col: str = "protein_name",
    label_col: str = "binds",
) -> list:
    """
    创建按蛋白质分层的 KFold 划分.

    与普通 StratifiedKFold 的区别:
      - 每个 fold 里三个蛋白质的正负样本比例都与总体一致
      - 使用 (protein_name + binds) 的组合作为分层标签

    这使得每个 fold 中:
      - BRD4 正样本: 与全局 BRD4 正样本比例一致
      - HSA 正样本: 与全局 HSA 正样本比例一致
      - sEH 正样本: 与全局 sEH 正样本比例一致

    Parameters
    ----------
    df : pd.DataFrame
        采样后的数据 (需包含 protein_name 和 binds 列).
    n_folds : int
        折数 (默认 5).
    random_state : int
        随机种子.
    protein_col : str
        蛋白质列名.
    label_col : str
        标签列名.

    Returns
    -------
    folds : list of tuple
        每个元素为 (train_indices, valid_indices).
    """
    # 创建组合分层标签: "BRD4_0", "BRD4_1", "HSA_0", "HSA_1", "sEH_0", "sEH_1"
    stratify_col = df[protein_col].astype(str) + "_" + df[label_col].astype(str)

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    folds: list = []
    for train_idx, valid_idx in skf.split(df, stratify_col):
        folds.append((train_idx, valid_idx))

    # 打印每折的分布
    print(f"\n[Sampler] 分层 KFold ({n_folds}-fold) 划分:")
    for fold_i, (train_idx, valid_idx) in enumerate(folds):
        train_sub = df.iloc[train_idx]
        valid_sub = df.iloc[valid_idx]

        n_train_pos = (train_sub[label_col] == 1).sum()
        n_valid_pos = (valid_sub[label_col] == 1).sum()

        print(f"  Fold {fold_i + 1}: "
              f"Train {len(train_idx):,} (正 {n_train_pos:,}), "
              f"Valid {len(valid_idx):,} (正 {n_valid_pos:,})")

    return folds
