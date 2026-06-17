"""
小规模端到端验证脚本
=====================

从 train.parquet 取前 N 万行, 跑完整的:
  数据加载 → 采样 → 特征工程 → 交叉验证训练 → 评估

目的: 在全量训练前快速验证全链路, 并看到有意义的 AUC 分数.
"""

import gc
import time
import os
import sys
import numpy as np
import pandas as pd

# 将项目根目录加入路径, 以便使用 `from src.xxx import ...`
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from src.config import (
    RANDOM_SEED,
    PROTEIN_NAMES,
    N_FOLDS,
    OUTPUT_DIR,
)


def main(
    n_rows: int = 2_000_000,      # 从 Parquet 读取的行数
    negative_ratio: int = 20,      # 正负比
    fingerprint_type: str = "morgan",
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> None:
    """
    小规模验证流程.

    Parameters
    ----------
    n_rows : int
        从 Parquet 读取的行数 (默认 200 万).
    negative_ratio : int
        负样本对正样本的倍数.
    fingerprint_type : str
        指纹类型.
    n_folds : int
        交叉验证折数.
    seed : int
        随机种子.
    """
    print("=" * 65)
    print(f"  BELKA 小规模验证")
    print(f"  读取行数: {n_rows:,}")
    print(f"  正负比:   1:{negative_ratio}")
    print(f"  指纹类型: {fingerprint_type}")
    print(f"  交叉验证: {n_folds}-fold")
    print("=" * 65)

    t0 = time.time()

    # ====================================================================
    # Step 1: 分块读取 + 边读边采样 (避免内存溢出)
    # ====================================================================
    print(f"\n[1/5] 分块流式读取 ({n_rows:,} 行) + 采样...")
    import pyarrow.parquet as pq
    from src.data_loader import _optimize_dtypes

    TRAIN_FILE = os.path.join(_script_dir, "data", "train.parquet")
    TRAIN_COLS = [
        "buildingblock1_smiles", "buildingblock2_smiles",
        "buildingblock3_smiles", "molecule_smiles",
        "protein_name", "binds",
    ]
    rng = np.random.RandomState(seed)

    pf = pq.ParquetFile(TRAIN_FILE)
    batch_size = 500_000  # 每块 50 万行
    pos_chunks, neg_chunks = [], []
    n_processed, n_p_found, n_n_kept, chunk_id = 0, 0, 0, 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=TRAIN_COLS):
        chunk_id += 1
        chunk_df = batch.to_pandas()
        chunk_df = _optimize_dtypes(chunk_df, convert_strings=False)

        # 分离正负
        df_pos = chunk_df[chunk_df["binds"] == 1].copy()
        df_neg = chunk_df[chunk_df["binds"] == 0].copy()
        nc_pos, nc_neg = len(df_pos), len(df_neg)

        if nc_pos > 0:
            pos_chunks.append(df_pos)
            n_p_found += nc_pos
            # 采样负样本
            target = nc_pos * negative_ratio
            if target < nc_neg:
                idx = rng.choice(nc_neg, size=target, replace=False)
                neg_chunks.append(df_neg.iloc[idx].copy())
                n_n_kept += target
            else:
                neg_chunks.append(df_neg)
                n_n_kept += nc_neg

        del chunk_df, df_pos, df_neg
        gc.collect()

        n_processed += batch.num_rows
        if n_processed >= n_rows:
            break

        if chunk_id % 5 == 0:
            print(f"  已处理 {n_processed / 1e6:.1f}M 行 | "
                  f"正样本累计 {n_p_found:,} | 负样本保留 {n_n_kept:,}")

    print(f"  读取完成: {n_processed:,} 行")
    print(f"  正样本提取: {n_p_found:,} ({100 * n_p_found / n_processed:.3f}%)")

    if n_p_found < 50:
        print(f"  WARNING: 正样本不足({n_p_found}), 请用 --rows 增大数据量"); return

    # 合并采样结果
    df_sampled = pd.concat(pos_chunks + neg_chunks, axis=0, ignore_index=True)
    df_sampled = _optimize_dtypes(df_sampled, convert_strings=True)
    df_sampled = df_sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    del pos_chunks, neg_chunks
    gc.collect()

    print(f"  采样后: {len(df_sampled):,} 行, "
          f"正 {int(df_sampled['binds'].sum()):,}, "
          f"负 {int((df_sampled['binds'] == 0).sum()):,}")
    print(f"  内存: {df_sampled.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

    # ====================================================================
    # Step 2: 采样已在上一步完成
    # ====================================================================
    print("\n[2/5] 采样已在上一步完成")

    # ====================================================================
    # Step 3: 特征工程
    # ====================================================================
    print("\n[3/5] 特征工程 (Morgan 指纹 + 理化性质 + 蛋白质编码)...")
    from src.featurizer import MolecularFeaturizer

    featurizer = MolecularFeaturizer(fingerprint_type=fingerprint_type)
    X = featurizer.fit_transform(df_sampled, protein_encoding="onehot")
    y = df_sampled["binds"].values.astype(np.float32)
    protein_labels = df_sampled["protein_name"].values

    print(f"  特征维度: {X.shape[1]}")
    print(f"  正样本:   {int(y.sum()):,}")
    print(f"  负样本:   {int(len(y) - y.sum()):,}")

    # ====================================================================
    # Step 4: 交叉验证训练
    # ====================================================================
    print(f"\n[4/5] 交叉验证训练 ({n_folds}-fold)...")
    from src.sampler import create_protein_stratified_folds
    from src.model import LightGBMTrainer
    from src.evaluation import evaluate_per_protein, print_cv_summary

    folds = create_protein_stratified_folds(
        df_sampled,
        n_folds=n_folds,
        random_state=seed,
    )

    fold_metrics = []
    cv_start = time.time()

    for fold_i, (train_idx, valid_idx) in enumerate(folds):
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]
        prot_val = protein_labels[valid_idx]

        # 训练
        trainer = LightGBMTrainer()
        trainer.fit(X_tr, y_tr, X_val, y_val)

        # 预测并评估
        y_pred = trainer.predict_proba(X_val)
        metrics = evaluate_per_protein(y_val, y_pred, prot_val)
        fold_metrics.append(metrics)

        # 每折结果
        print(f"  Fold {fold_i + 1}:")
        for prot in PROTEIN_NAMES:
            auc = metrics.get(f"{prot}_auc", np.nan)
            print(f"    {prot:6s} AUC: {auc:.6f}")
        print(f"    Mean  AUC: {metrics.get('mean_auc', np.nan):.6f}")
        print()

        del X_tr, X_val, y_tr, y_val
        gc.collect()

    cv_elapsed = (time.time() - cv_start) / 60
    print(f"  交叉验证耗时: {cv_elapsed:.1f} 分钟")

    # ====================================================================
    # Step 5: 汇总
    # ====================================================================
    print("\n[5/5] 结果汇总")
    print_cv_summary(fold_metrics, label="Cross-Validation")

    total_elapsed = (time.time() - t0) / 60
    print(f"\n总耗时: {total_elapsed:.1f} 分钟")
    print("===== 小规模验证完成! =====")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BELKA 小规模验证")
    parser.add_argument("--rows", type=int, default=2_000_000,
                        help="读取行数 (默认 200 万)")
    parser.add_argument("--ratio", type=int, default=20,
                        help="负样本比例 (默认 20)")
    parser.add_argument("--folds", type=int, default=5,
                        help="交叉验证折数 (默认 5)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="随机种子")
    args = parser.parse_args()

    main(
        n_rows=args.rows,
        negative_ratio=args.ratio,
        n_folds=args.folds,
        seed=args.seed,
    )
