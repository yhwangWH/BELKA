"""
数据加载模块
=============

负责:
  1. 按需从 Parquet 文件读取训练/测试数据
  2. 内存优化: 使用合适的数据类型降低内存占用
  3. 数据基本信息打印与分布统计
"""

import os
import gc
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

from .config import (
    TRAIN_FILE,
    TEST_FILE,
    SAMPLE_SUB,
    TRAIN_COLUMNS,
    TEST_COLUMNS,
    PROTEIN_NAMES,
    NEGATIVE_RATIO,
    MAX_POSITIVE_SAMPLES,
    MAX_TOTAL_TRAIN_SAMPLES,
    DATA_CHUNK_SIZE,
    RANDOM_SEED,
)


def load_train_data(
    filepath: str = TRAIN_FILE,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    use_dtype_optimization: bool = True,
) -> pd.DataFrame:
    """
    从 Parquet 文件加载训练数据, 并进行内存优化.

    Parameters
    ----------
    filepath : str
        训练数据 Parquet 文件路径.
    columns : list of str, optional
        需要读取的列名列表. 若为 None, 则读取所有列.
    nrows : int, optional
        限制读取行数 (用于快速测试). 若为 None, 则读取全部数据.
    use_dtype_optimization : bool
        是否对数据类型进行优化以降低内存占用. 默认 True.

    Returns
    -------
    df : pd.DataFrame
        训练数据 DataFrame.

    Notes
    -----
    全量数据约 2.95 亿行 (~0.3B). 强烈建议:
      1. 使用 columns 参数只读取需要的列
      2. 使用类别编码存储 SMILES 字符串
      3. 使用 int8 存储二值标签
    """
    print(f"[DataLoader] 正在读取训练数据: {filepath}")
    print(f"  文件大小: {os.path.getsize(filepath) / (1024**3):.2f} GB")

    if columns is None:
        columns = TRAIN_COLUMNS

    # 使用 pyarrow 引擎读取 Parquet
    df = pd.read_parquet(filepath, columns=columns)

    if nrows is not None:
        df = df.iloc[:nrows].copy()
        print(f"  只加载了前 {nrows} 行 (快速测试模式)")

    # 内存优化
    if use_dtype_optimization:
        df = _optimize_dtypes(df)

    _print_data_info(df, label="训练数据")
    return df


def load_test_data(
    filepath: str = TEST_FILE,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    从 Parquet 文件加载测试数据.

    Parameters
    ----------
    filepath : str
        测试数据 Parquet 文件路径.
    columns : list of str, optional
        需要读取的列名列表. 若为 None, 则读取所有列.
    nrows : int, optional
        限制读取行数. 若为 None, 则读取全部数据.

    Returns
    -------
    df : pd.DataFrame
        测试数据 DataFrame.
    """
    print(f"[DataLoader] 正在读取测试数据: {filepath}")
    print(f"  文件大小: {os.path.getsize(filepath) / (1024**3):.2f} GB")

    if columns is None:
        columns = TEST_COLUMNS

    df = pd.read_parquet(filepath, columns=columns)

    if nrows is not None:
        df = df.iloc[:nrows].copy()
        print(f"  只加载了前 {nrows} 行 (快速测试模式)")

    # 测试集也需要内存优化
    for col in df.select_dtypes(include=["int64"]).columns:
        col_max = df[col].max()
        if col_max < 2_147_483_648:
            df[col] = df[col].astype("int32")
    for col in df.columns:
        if str(df[col].dtype) in ("string[pyarrow]", "object"):
            df[col] = df[col].astype(str).astype("category")

    print(f"[DataLoader] 测试数据: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    return df


def load_sample_submission(filepath: str = SAMPLE_SUB) -> pd.DataFrame:
    """
    加载 Kaggle 官方提交模板.

    Parameters
    ----------
    filepath : str
        sample_submission.csv 文件路径.

    Returns
    -------
    df : pd.DataFrame
        包含 'id' 和 'binds' 列的模板 DataFrame.
    """
    print(f"[DataLoader] 正在读取提交模板: {filepath}")
    # sample_submission.csv 可能较大, 用 chunk 方式读第一行确认格式
    # 全量读取用于后续写入时确保行数匹配
    df = pd.read_csv(filepath, dtype={"id": "int32", "binds": "float32"})
    print(f"  提交模板: {df.shape[0]:,} 条记录")
    print(f"  列名: {df.columns.tolist()}")
    return df


# ============================================================================
# 内部工具函数
# ============================================================================

def _optimize_dtypes(df: pd.DataFrame, convert_strings: bool = True) -> pd.DataFrame:
    """
    对 DataFrame 的列进行数据类型优化以降低内存占用.

    优化策略:
      - int64  → 按数据范围转 int32/int16/int8
      - float64 → float32 (对比赛精度足够)
      - object/ArrowString → category (对低基数的字符串列)
      - 标签列 binds → int8 (0/1 足够)

    Parameters
    ----------
    df : pd.DataFrame
        原始 DataFrame.
    convert_strings : bool
        是否转换字符串列为 category. 分块模式下可关闭.

    Returns
    -------
    df : pd.DataFrame
        类型优化后的 DataFrame.
    """
    initial_mem = df.memory_usage(deep=True).sum() / (1024**2)

    # --- 整数列降级 ---
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype("int8")
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype("int16")
        elif col_min >= -2_147_483_648 and col_max <= 2_147_483_647:
            df[col] = df[col].astype("int32")

    # --- 浮点列降级 ---
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    # --- 字符串列转为 category (SMILES 重复度高, 收益大) ---
    if convert_strings:
        for col in df.select_dtypes(include=["object"]).columns:
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if unique_ratio < 0.5:
                df[col] = df[col].astype("category")

        # 处理 ArrowDtype 字符串列 (pandas 2.x + pyarrow)
        for col in df.columns:
            if str(df[col].dtype) == "string[pyarrow]":
                unique_ratio = df[col].nunique() / max(len(df), 1)
                if unique_ratio < 0.5:
                    # 先转为 Python string 再 category
                    df[col] = df[col].astype(str).astype("category")

    # 手动确保 binds 列是 int8
    if "binds" in df.columns:
        df["binds"] = df["binds"].astype("int8")

    optimized_mem = df.memory_usage(deep=True).sum() / (1024**2)
    reduction = 100 * (1 - optimized_mem / max(initial_mem, 0.01))
    if reduction > 0:
        print(f"  内存优化: {initial_mem:.1f} MB → {optimized_mem:.1f} MB "
              f"(降低 {reduction:.1f}%)")

    return df


def load_and_sample_train_chunked(
    filepath: str = TRAIN_FILE,
    columns: Optional[List[str]] = None,
    negative_ratio: int = NEGATIVE_RATIO,
    max_positive_per_protein: int = MAX_POSITIVE_SAMPLES,
    max_total_samples: int = MAX_TOTAL_TRAIN_SAMPLES,
    chunk_size: int = DATA_CHUNK_SIZE,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    分块流式读取 Parquet, 边读边采样, 定期合并缓冲避免 OOM.

    关键改进 (v2):
      - 每 merge_interval 个块合并一次缓冲区, 防止 pos_chunks/neg_chunks
        列表无限膨胀 (原实现会累积全部 2.95 亿行的采样结果).
      - 尊重 MAX_POSITIVE_SAMPLES 上限.
      - 达到 MAX_TOTAL_TRAIN_SAMPLES 后提前退出.

    Returns
    -------
    sampled_df : pd.DataFrame
        采样后的训练数据 (大小可控, ~百万行级别).
    """
    import pyarrow.parquet as pq

    if columns is None:
        columns = TRAIN_COLUMNS

    merge_interval = 5  # 每 5 个 chunk (~10M 行) 合并一次缓冲区

    print(f"[DataLoader] 分块流式读取 + 采样 (v2 内存优化): {filepath}")
    print(f"  负样本比例: 1:{negative_ratio}")
    print(f"  每蛋白最大正样本: {max_positive_per_protein:,}")
    print(f"  总样本硬上限: {max_total_samples:,}")
    print(f"  块大小: {chunk_size:,} 行/块, 每 {merge_interval} 块合并缓冲")

    pf = pq.ParquetFile(filepath)
    total_rows = pf.metadata.num_rows
    print(f"  总行数: {total_rows:,}")

    rng = np.random.RandomState(random_state)

    final_pos: List[pd.DataFrame] = []
    final_neg: List[pd.DataFrame] = []
    pos_chunks: List[pd.DataFrame] = []
    neg_chunks: List[pd.DataFrame] = []

    n_processed = 0
    n_pos_found = 0
    chunk_id = 0
    early_stop = False

    for batch in pf.iter_batches(
        batch_size=chunk_size,
        columns=columns,
    ):
        if early_stop:
            break

        chunk_id += 1
        df_chunk = batch.to_pandas()
        df_chunk = _optimize_dtypes(df_chunk, convert_strings=False)

        pos_mask = df_chunk["binds"] == 1
        df_pos = df_chunk[pos_mask].copy()
        df_neg = df_chunk[~pos_mask].copy()

        n_chunk_pos = len(df_pos)
        n_chunk_neg = len(df_neg)

        if n_chunk_pos > 0:
            pos_chunks.append(df_pos)
            n_pos_found += n_chunk_pos

        if n_chunk_neg > 0:
            target_neg = n_chunk_pos * negative_ratio
            if target_neg >= n_chunk_neg:
                neg_chunks.append(df_neg)
            else:
                chosen = rng.choice(n_chunk_neg, size=target_neg, replace=False)
                neg_chunks.append(df_neg.iloc[chosen].copy())
                del df_neg
                gc.collect()

        del df_chunk, df_pos
        gc.collect()
        n_processed += batch.num_rows

        # ---- 定期合并缓冲区, 防止内存膨胀 ----
        if chunk_id % merge_interval == 0:
            pct = 100 * n_processed / total_rows
            buf_mb = _estimate_buf_memory(pos_chunks, neg_chunks)
            print(f"  进度: {n_processed / 1e6:.0f}M/{total_rows / 1e6:.0f}M "
                  f"({pct:.1f}%) | 正: {n_pos_found:,} | 缓冲内存: {buf_mb:.0f} MB")

            _merge_buffers(pos_chunks, neg_chunks, final_pos, final_neg,
                           max_positive_per_protein, random_state)

            total_so_far = sum(len(df) for df in final_pos) + sum(len(df) for df in final_neg)
            if total_so_far >= max_total_samples:
                print(f"  [DataLoader] 已达总样本上限 {max_total_samples:,}, 提前结束")
                early_stop = True

    # 处理残留块
    _merge_buffers(pos_chunks, neg_chunks, final_pos, final_neg,
                   max_positive_per_protein, random_state)

    # 最终合并
    print(f"[DataLoader] 最终合并 ({len(final_pos)} 正块 + {len(final_neg)} 负块)...")
    all_pos = pd.concat(final_pos, axis=0, ignore_index=True) if final_pos else pd.DataFrame()
    all_neg = pd.concat(final_neg, axis=0, ignore_index=True) if final_neg else pd.DataFrame()
    del final_pos, final_neg, pos_chunks, neg_chunks
    gc.collect()

    sampled_df = pd.concat([all_pos, all_neg], axis=0, ignore_index=True)
    del all_pos, all_neg
    gc.collect()

    sampled_df = _optimize_dtypes(sampled_df, convert_strings=True)
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 硬截断 (保险)
    if len(sampled_df) > max_total_samples:
        print(f"  [DataLoader] 硬截断: {len(sampled_df):,} → {max_total_samples:,}")
        sampled_df = sampled_df.iloc[:max_total_samples].reset_index(drop=True)

    _print_data_info(sampled_df, label="采样后训练数据")
    return sampled_df


def _merge_buffers(
    pos_chunks: list,
    neg_chunks: list,
    final_pos: list,
    final_neg: list,
    max_per_protein: int,
    random_state: int,
) -> None:
    """合并当前缓冲窗口到最终存储, 并对正样本按蛋白质限流."""
    if pos_chunks:
        merged = pd.concat(pos_chunks, axis=0, ignore_index=True)
        merged = _cap_positives_per_protein(merged, max_per_protein, random_state)
        final_pos.append(merged)
        pos_chunks.clear()
    if neg_chunks:
        final_neg.append(pd.concat(neg_chunks, axis=0, ignore_index=True))
        neg_chunks.clear()
    gc.collect()


def _cap_positives_per_protein(
    df: pd.DataFrame,
    max_per_protein: int,
    random_state: int,
) -> pd.DataFrame:
    """对每个蛋白质的正样本设上限, 超出部分随机丢弃."""
    if len(df) == 0:
        return df
    capped = []
    for prot in PROTEIN_NAMES:
        sub = df[df["protein_name"] == prot]
        if len(sub) > max_per_protein:
            sub = sub.sample(n=max_per_protein, random_state=random_state)
        if len(sub) > 0:
            capped.append(sub)
    result = pd.concat(capped, axis=0, ignore_index=True) if capped else df.iloc[:0].copy()
    del capped
    gc.collect()
    return result


def _estimate_buf_memory(
    pos_chunks: list, neg_chunks: list
) -> float:
    """估算当前缓冲区内存占用 (MB) — 轻量化版本."""
    total = 0.0
    for c in pos_chunks[-3:]:
        try:
            total += c.memory_usage(deep=True).sum()
        except Exception:
            pass
    for c in neg_chunks[-3:]:
        try:
            total += c.memory_usage(deep=True).sum()
        except Exception:
            pass
    return total / (1024**2)


def _print_data_info(df: pd.DataFrame, label: str = "") -> None:
    """
    打印 DataFrame 基本信息和分布统计.

    Parameters
    ----------
    df : pd.DataFrame
        数据.
    label : str
        数据标签 (用于输出区分).
    """
    print(f"\n{'='*60}")
    print(f"  {label} 概览")
    print(f"{'='*60}")
    print(f"  行数: {df.shape[0]:,}")
    print(f"  列数: {df.shape[1]}")
    print(f"  列名: {df.columns.tolist()}")
    print(f"  内存: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    print(f"  数据类型:\n{df.dtypes.value_counts().to_string()}")

    # 如果包含标签列, 打印正负样本分布
    if "binds" in df.columns:
        print(f"\n  --- 标签分布 ---")
        total = len(df)
        pos = df["binds"].sum()
        neg = total - pos
        print(f"  正样本 (binds=1): {pos:,} ({100 * pos / total:.4f}%)")
        print(f"  负样本 (binds=0): {neg:,} ({100 * neg / total:.4f}%)")
        print(f"  正负比: 1:{neg / pos:.1f}" if pos > 0 else "  正负比: 无限大 (无正样本)")

        # 每个蛋白质的分布
        if "protein_name" in df.columns:
            print(f"\n  --- 各蛋白质标签分布 ---")
            for protein in PROTEIN_NAMES:
                subset = df[df["protein_name"] == protein]
                if len(subset) == 0:
                    continue
                p_pos = subset["binds"].sum()
                p_total = len(subset)
                print(f"    {protein:6s}: 正 {p_pos:>8,} / 总 {p_total:>10,} "
                      f"({100 * p_pos / p_total:.4f}%)")
    print()
