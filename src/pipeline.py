"""
主训练流程 (Pipeline)
======================

端到端的第一阶段基线训练流程, 串联所有模块:

  Step 1: 数据加载 (data_loader)
  Step 2: 分层采样 (sampler)
  Step 3: 特征工程 (featurizer)
  Step 4: 交叉验证训练 + 评估 (model + evaluation)
  Step 5: 测试集预测 + 生成提交文件 (submission)

支持两种运行模式:
  - 全量模式: 使用全部训练数据 (train.parquet), 完整训练
  - 快速模式: 使用 300 行小数据 (train_300.csv), 快速验证流程

Usage:
  python -m src.pipeline          # 快速模式 (300行)
  python -m src.pipeline --full   # 全量模式
"""

import argparse
import gc
import time
import os
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from .config import (
    TRAIN_FILE,
    TEST_FILE,
    SAMPLE_SUB,
    PROTEIN_NAMES,
    N_FOLDS,
    RANDOM_SEED,
    OUTPUT_DIR,
    MODEL_DIR,
    SUBMISSION_DIR,
    VERBOSE_EVAL,
    ensure_output_dirs,
)
from .data_loader import (
    load_train_data,
    load_test_data,
    load_sample_submission,
    load_and_sample_train_chunked,
)
from .featurizer import MolecularFeaturizer
from .sampler import stratified_sample, create_protein_stratified_folds
from .model import LightGBMTrainer, optimize_lgb_hyperparams
from .evaluation import (
    evaluate_per_protein,
    print_cv_summary,
    plot_training_curves,
)
from .submission import (
    create_submission,
    collect_oof_predictions,
    compute_oof_score,
)


class BELKAPipeline:
    """
    BELKA 比赛第一阶段完整流程编排器.

    Attributes
    ----------
    quick_mode : bool
        是否为快速验证模式 (使用 300 行小数据).
    use_optuna : bool
        是否进行 Optuna 超参数搜索.
    featurizer : MolecularFeaturizer
        特征提取器实例.
    models : list of LightGBMTrainer
        各折训练的模型.
    oof_df : pd.DataFrame
        包含 OOF 预测的 DataFrame.
    """

    def __init__(
        self,
        quick_mode: bool = True,
        use_optuna: bool = False,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        """
        Parameters
        ----------
        quick_mode : bool
            True → 使用 train_300.csv 快速验证 (< 1分钟).
            False → 使用 train.parquet 全量数据.
        use_optuna : bool
            是否使用 Optuna 搜索超参数.
        random_seed : int
            随机种子.
        """
        self.quick_mode = quick_mode
        self.use_optuna = use_optuna
        self.random_seed = random_seed

        # 初始化组件 (稍后在 run() 中填充)
        self.featurizer: Optional[MolecularFeaturizer] = None
        self.models: List[LightGBMTrainer] = []
        self.oof_df: Optional[pd.DataFrame] = None
        self.fold_metrics: List[Dict[str, float]] = []

        self.run_name = "quick_test" if quick_mode else "full_train"
        print(f"\n{'#'*60}")
        print(f"#  BELKA Pipeline - Phase 1 Baseline")
        print(f"#  模式: {'快速验证 (300行)' if quick_mode else '全量训练'}")
        print(f"#  Optuna: {'启用' if use_optuna else '禁用 (使用默认参数)'}")
        print(f"#  随机种子: {random_seed}")
        print(f"{'#'*60}\n")

    # =========================================================================
    # 主入口
    # =========================================================================

    def run(self) -> Dict[str, float]:
        """
        执行完整的训练与预测流程.

        Returns
        -------
        results : dict
            包含最终评估指标的字典.
        """
        t_start = time.time()

        # ---- 确保输出目录存在 (Kaggle 环境下指向 /kaggle/working) ----
        ensure_output_dirs()

        # ---- Step 1: 加载数据 ----
        print("\n" + "=" * 60)
        print("  Step 1: 数据加载")
        print("=" * 60)

        df_train = self._load_train_data()
        df_test = self._load_test_data()

        # ---- Step 2: 分层采样 (全量模式已在加载时完成) ----
        if not self.quick_mode:
            # 全量模式: 分块加载时已同时完成采样, df_train 即采样后数据
            print("\n" + "=" * 60)
            print("  Step 2: 采样已在上一步 (分块加载) 中完成")
            print("=" * 60)
            df_sampled = df_train  # df_train 已是采样后数据
        else:
            print("\n" + "=" * 60)
            print("  Step 2: 分层采样 (正负比控制)")
            print("=" * 60)
            df_sampled = self._sample_data(df_train)

        # ---- Step 3: 特征工程 ----
        print("\n" + "=" * 60)
        print("  Step 3: 特征工程 (Morgan指纹 + 理化性质 + 蛋白质编码)")
        print("=" * 60)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] 训练集行数: {len(df_sampled):,}")
        print(f"  ⏳ 特征工程是最耗时的步骤, 请耐心等待进度输出...")

        t_feat = time.time()
        X, y, protein_labels = self._extract_features(df_sampled)
        feat_elapsed = time.time() - t_feat
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] 训练集特征提取完成 "
              f"(耗时 {feat_elapsed:.1f}s)")

        print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] 开始提取测试集特征...")
        X_test, test_ids, test_protein_labels = self._extract_test_features(df_test)

        # 释放原始训练数据内存 (保留 df_sampled 供交叉验证分层用)
        del df_train
        gc.collect()

        # ---- Step 4: 交叉验证训练 ----
        print("\n" + "=" * 60)
        print("  Step 4: 交叉验证训练 (5-fold, 按蛋白质分层)")
        print("=" * 60)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] 开始 {N_FOLDS}-fold 交叉验证...")
        print(f"  每折训练约需数分钟, LightGBM 会每 {VERBOSE_EVAL} 轮打印一次 AUC\n")

        self._run_cross_validation(X, y, protein_labels, df_sampled)

        # 交叉验证完成后释放 df_sampled
        del df_sampled
        gc.collect()

        # ---- Step 5: 测试集预测 ----
        print("\n" + "=" * 60)
        print("  Step 5: 测试集预测与提交生成")
        print("=" * 60)

        test_preds = self._predict_test(X_test)
        self._generate_submission(test_ids, test_preds)

        # ---- 清理特征文件 ----
        if self.featurizer is not None:
            self.featurizer.cleanup_memmap()

        # ---- 总结 ----
        elapsed = time.time() - t_start
        print(f"\n{'#'*60}")
        print(f"#  Pipeline 完成! 总耗时: {elapsed / 60:.1f} 分钟")
        print(f"{'#'*60}")

        return self._final_summary()

    # =========================================================================
    # Step 1: 数据加载
    # =========================================================================

    def _load_train_data(self) -> pd.DataFrame:
        """加载训练数据 (快速模式用 CSV, 全量模式用 Parquet)."""
        if self.quick_mode:
            # 快速模式: 从 train_300.csv 读取
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "train_300.csv"
            )
            print(f"[Pipeline] 快速模式: 读取 {csv_path}")
            df = pd.read_csv(csv_path)
            # 模拟全量数据的内存优化
            from .data_loader import _optimize_dtypes
            df = _optimize_dtypes(df)
        else:
            # 全量模式: 分块流式读取 + 边读边采样 (避免 2.95 亿行 OOM)
            df = load_and_sample_train_chunked()
        return df

    def _load_test_data(self) -> pd.DataFrame:
        """加载测试数据."""
        if self.quick_mode:
            # 快速模式: 用 train_300 的前 50 行模拟测试集
            print("[Pipeline] 快速模式: 使用训练数据前 50 行模拟测试集")
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "train_300.csv"
            )
            df = pd.read_csv(csv_path, nrows=50)
            # 测试集不应该有 binds 列, 但我们保留用于快速验证
        else:
            df = load_test_data()
        return df

    # =========================================================================
    # Step 2: 分层采样
    # =========================================================================

    def _sample_data(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """执行分层采样."""
        if self.quick_mode:
            print("[Pipeline] 快速模式: 跳过采样 (数据量小)")
            return df_train.copy()

        df_sampled = stratified_sample(df_train)
        return df_sampled

    # =========================================================================
    # Step 3: 特征工程
    # =========================================================================

    def _extract_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从 DataFrame 提取特征矩阵和标签.

        Returns
        -------
        X : np.ndarray
            特征矩阵.
        y : np.ndarray
            标签.
        protein_labels : np.ndarray
            蛋白质名称 (用于 per-protein 评估).
        """
        self.featurizer = MolecularFeaturizer()

        X = self.featurizer.fit_transform(df, protein_encoding="onehot")
        y = df["binds"].values.astype(np.float32)
        protein_labels = df["protein_name"].values

        return X, y, protein_labels

    def _extract_test_features(
        self, df_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取测试集特征."""
        if self.featurizer is None:
            raise RuntimeError("请先调用 _extract_features() 初始化 featurizer")

        print(f"\n[Pipeline] 提取测试集特征...")
        X_test = self.featurizer.transform_test(df_test, protein_encoding="onehot")

        test_ids = df_test["id"].values if "id" in df_test.columns else np.arange(len(df_test))
        test_protein_labels = df_test["protein_name"].values

        return X_test, test_ids, test_protein_labels

    # =========================================================================
    # Step 4: 交叉验证
    # =========================================================================

    def _run_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protein_labels: np.ndarray,
        df_original: pd.DataFrame,
    ) -> None:
        """
        执行按蛋白质分层的 5-fold 交叉验证.

        流程:
          1. 创建分层 KFold 划分
          2. 对每折:
             a. 可选 Optuna 超参数搜索
             b. 训练 LightGBM 模型
             c. 评估验证集 per-protein AUC
             d. 保存模型
          3. 汇总所有折的评估结果
        """
        folds = create_protein_stratified_folds(df_original)

        fold_predictions: list = []
        self.fold_metrics = []
        self.models = []

        for fold_i, (train_idx, valid_idx) in enumerate(folds):
            print(f"\n{'='*50}")
            print(f"  Fold {fold_i + 1}/{len(folds)}")
            print(f"{'='*50}")

            # 划分数据
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            prot_valid = protein_labels[valid_idx]

            # Optuna 超参数搜索 (可选)
            if self.use_optuna and fold_i == 0:
                print("\n[Pipeline] Optuna 超参数搜索 (仅第一折)...")
                best_params = optimize_lgb_hyperparams(
                    X_train, y_train, X_valid, y_valid
                )
                trainer_params = best_params
            else:
                from .config import LIGHTGBM_PARAMS
                trainer_params = LIGHTGBM_PARAMS.copy()

            # 训练
            trainer = LightGBMTrainer(params=trainer_params)
            trainer.fit(X_train, y_train, X_valid, y_valid)

            # 预测验证集
            y_pred = trainer.predict_proba(X_valid)

            # 评估
            metrics = evaluate_per_protein(y_valid, y_pred, prot_valid)
            self.fold_metrics.append(metrics)

            # 打印每折结果
            for protein in PROTEIN_NAMES:
                auc = metrics.get(f"{protein}_auc", np.nan)
                print(f"    {protein:6s} AUC: {auc:.6f}")
            print(f"    Mean  AUC: {metrics.get('mean_auc', np.nan):.6f}")

            # 记录 OOF 预测
            fold_predictions.append({
                "valid_idx": valid_idx,
                "y_valid": y_valid,
                "pred_prob": y_pred,
                "protein": prot_valid,
            })

            # 保存模型
            model_path = os.path.join(
                MODEL_DIR, f"lgb_phase1_fold{fold_i + 1}_{self.run_name}.pkl"
            )
            trainer.save(model_path)
            self.models.append(trainer)

            # 清理
            del X_train, X_valid, y_train, y_valid
            gc.collect()

        # 汇总交叉验证结果
        print_cv_summary(self.fold_metrics, label="Cross-Validation")

        # 收集并计算 OOF 分数
        self.oof_df = collect_oof_predictions(fold_predictions, df_original)
        compute_oof_score(self.oof_df)

    # =========================================================================
    # Step 5: 测试集预测与提交
    # =========================================================================

    def _predict_test(self, X_test: np.ndarray) -> np.ndarray:
        """
        使用所有折模型进行测试集预测 (取平均).

        策略: 对每个模型分别预测, 然后取均值.
        这相当于简单的模型集成 (bagging across folds).

        Parameters
        ----------
        X_test : np.ndarray
            测试集特征.

        Returns
        -------
        test_preds : np.ndarray
            平均预测概率.
        """
        print(f"[Pipeline] 使用 {len(self.models)} 个模型进行集成预测...")

        all_preds = np.zeros((len(X_test), len(self.models)), dtype=np.float32)

        for i, trainer in enumerate(self.models):
            y_pred = trainer.predict_proba(X_test)
            all_preds[:, i] = y_pred
            print(f"  Model {i + 1}: pred range [{y_pred.min():.6f}, "
                  f"{y_pred.max():.6f}], mean={y_pred.mean():.6f}")

        # 取均值
        test_preds = all_preds.mean(axis=1)

        print(f"  Ensemble: pred range [{test_preds.min():.6f}, "
              f"{test_preds.max():.6f}], mean={test_preds.mean():.6f}")

        return test_preds

    def _generate_submission(
        self,
        test_ids: np.ndarray,
        test_preds: np.ndarray,
    ) -> Optional[str]:
        """生成 Kaggle 提交文件."""
        from .submission import clip_predictions

        output_name = f"submission_phase1_{self.run_name}.csv"

        if self.quick_mode:
            # 快速模式: 直接生成小型提交文件 (test 来自 train_300 前50行)
            print("[Pipeline] 快速模式: 生成小型提交文件...")
            test_preds = clip_predictions(test_preds)
            submission = pd.DataFrame({
                "id": test_ids,
                "binds": test_preds,
            })
            output_path = os.path.join(SUBMISSION_DIR, output_name)
            submission.to_csv(output_path, index=False)
            print(f"[Submission] 提交文件已生成: {output_path}")
            print(f"  行数: {len(submission):,}")
            print(f"  预测范围: [{test_preds.min():.6f}, {test_preds.max():.6f}]")
            print(f"  预测均值: {test_preds.mean():.6f}")
            return output_path

        output_path = create_submission(
            test_ids=test_ids,
            test_preds=test_preds,
            sample_sub_path=SAMPLE_SUB,
            output_name=output_name,
        )
        return output_path

    # =========================================================================
    # 总结
    # =========================================================================

    def _final_summary(self) -> Dict[str, float]:
        """最终结果汇总."""
        if not self.fold_metrics:
            return {}

        from .evaluation import cross_validation_summary
        summary = cross_validation_summary(self.fold_metrics)

        print(f"\n{'='*60}")
        print(f"  最终结果汇总")
        print(f"{'='*60}")
        print(f"  使用 {self.run_name} 模式")

        if "mean_auc" in summary:
            mean = summary["mean_auc"]["mean"]
            std = summary["mean_auc"]["std"]
            print(f"  5-Fold CV Mean AUC: {mean:.6f} ± {std:.6f}")

        for protein in PROTEIN_NAMES:
            key = f"{protein}_auc"
            if key in summary:
                m = summary[key]["mean"]
                s = summary[key]["std"]
                print(f"  {protein:6s} AUC: {m:.6f} ± {s:.6f}")

        print(f"{'='*60}\n")

        return {
            k: v["mean"] for k, v in summary.items() if "mean_auc" not in k.lower()
        }


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    """命令行入口: 支持 --full 和 --optuna 参数."""
    parser = argparse.ArgumentParser(
        description="BELKA Phase 1 Baseline Pipeline"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="全量数据训练模式 (默认使用 300 行快速验证)",
    )
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="启用 Optuna 超参数搜索",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"随机种子 (默认 {RANDOM_SEED})",
    )
    args = parser.parse_args()

    pipeline = BELKAPipeline(
        quick_mode=not args.full,
        use_optuna=args.optuna,
        random_seed=args.seed,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
