"""
模型模块
========

封装 LightGBM 和 XGBoost 模型训练与推理:

  - LightGBM 基线训练 (含 class_weight 和 scale_pos_weight)
  - XGBoost 基线训练 (备用方案)
  - Optuna 超参数自动搜索
  - 模型保存/加载

Usage:
  >>> trainer = LightGBMTrainer(params)
  >>> trainer.fit(X_train, y_train, X_valid, y_valid)
  >>> probs = trainer.predict_proba(X_test)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .config import (
    LIGHTGBM_PARAMS,
    XGBOOST_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    LIGHTGBM_PARAM_SPACE,
    RANDOM_SEED,
    MODEL_DIR,
)


class LightGBMTrainer:
    """
    LightGBM 分类器训练器.

    内置类别不平衡处理:
      - is_unbalance=True: 自动按正负样本比例调整权重
      - scale_pos_weight=200: 额外放大正样本权重

    支持:
      - 早停 (early stopping)
      - 验证集监控
      - 模型保存/加载
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
        verbose_eval: int = VERBOSE_EVAL,
    ) -> None:
        """
        Parameters
        ----------
        params : dict, optional
            LightGBM 参数. 若为 None, 使用 config 中的默认参数.
        early_stopping_rounds : int
            早停轮数.
        verbose_eval : int
            每多少轮打印一次评估指标.
        """
        if params is None:
            params = LIGHTGBM_PARAMS.copy()
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval

        self.model_: Optional[lgb.Booster] = None
        self.evals_result_: Dict[str, Dict[str, list]] = {}
        self.best_iteration_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        categorical_feature: str = "auto",
    ) -> "LightGBMTrainer":
        """
        训练 LightGBM 模型.

        Parameters
        ----------
        X_train : np.ndarray
            训练特征.
        y_train : np.ndarray
            训练标签.
        X_valid : np.ndarray
            验证特征.
        y_valid : np.ndarray
            验证标签.
        categorical_feature : str or list
            类别特征索引, "auto" 则自动检测.

        Returns
        -------
        self
        """
        # 构建 LightGBM Dataset
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_feature,
        )
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=dtrain,
        )

        # 训练参数
        fit_params = self.params.copy()
        n_estimators = fit_params.pop("n_estimators", 2000)

        # 类别不平衡处理:
        #   使用 scale_pos_weight 放大正样本权重 (~正负比倒数)
        #   注意: 不能同时使用 is_unbalance=True
        fit_params.setdefault("scale_pos_weight", 100)
        # 如果有 is_unbalance 则移除 (与 scale_pos_weight 互斥)
        fit_params.pop("is_unbalance", None)

        evals_result = {}

        self.model_ = lgb.train(
            params=fit_params,
            train_set=dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.record_evaluation(evals_result),
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(period=self.verbose_eval),
            ],
        )

        self.evals_result_ = evals_result
        self.best_iteration_ = self.model_.best_iteration
        self.feature_importances_ = (
            self.model_.feature_importance(importance_type="gain")
        )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测正类概率.

        Parameters
        ----------
        X : np.ndarray
            特征矩阵.

        Returns
        -------
        probs : np.ndarray, shape (n_samples,)
            预测概率值.
        """
        if self.model_ is None:
            raise RuntimeError("模型尚未训练, 请先调用 fit()")
        return self.model_.predict(X)

    def save(self, filepath: str) -> None:
        """保存模型到文件 (使用 pickle)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model_, f)
        print(f"[Model] 模型已保存至: {filepath}")

    def load(self, filepath: str) -> "LightGBMTrainer":
        """从文件加载模型."""
        with open(filepath, "rb") as f:
            self.model_ = pickle.load(f)
        print(f"[Model] 模型已加载自: {filepath}")
        return self

    def get_feature_importance_df(
        self, feature_names: Optional[list] = None
    ) -> pd.DataFrame:
        """
        获取特征重要性 DataFrame (按 gain 排序).

        Parameters
        ----------
        feature_names : list of str, optional
            特征名列表. 如果提供了 featurizer.feature_names_, 会有列名.

        Returns
        -------
        df : pd.DataFrame
            包含 feature, importance 两列的 DataFrame, 降序排列.
        """
        importances = self.feature_importances_
        if importances is None:
            raise RuntimeError("模型尚未训练, 无特征重要性")

        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(len(importances))]

        df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df


class XGBoostTrainer:
    """
    XGBoost 分类器训练器 (备用方案).

    使用方式与 LightGBMTrainer 类似.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
        verbose_eval: int = VERBOSE_EVAL,
    ) -> None:
        if params is None:
            params = XGBOOST_PARAMS.copy()
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval

        self.model_: Optional[xgb.Booster] = None
        self.evals_result_: Dict[str, Dict[str, list]] = {}
        self.best_iteration_: int = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
    ) -> "XGBoostTrainer":
        """
        训练 XGBoost 模型.

        Parameters
        ----------
        X_train, y_train : 训练数据.
        X_valid, y_valid : 验证数据.

        Returns
        -------
        self
        """
        fit_params = self.params.copy()
        n_estimators = fit_params.pop("n_estimators", 2000)

        evals_result = {}

        self.model_ = xgb.train(
            params=fit_params,
            dtrain=xgb.DMatrix(X_train, label=y_train),
            num_boost_round=n_estimators,
            evals=[
                (xgb.DMatrix(X_train, label=y_train), "train"),
                (xgb.DMatrix(X_valid, label=y_valid), "valid"),
            ],
            evals_result=evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
        )

        self.evals_result_ = evals_result
        self.best_iteration_ = self.model_.best_iteration
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测正类概率."""
        if self.model_ is None:
            raise RuntimeError("模型尚未训练, 请先调用 fit()")
        return self.model_.predict(xgb.DMatrix(X))


# ============================================================================
# Optuna 超参数搜索
# ============================================================================

def optimize_lgb_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    使用 Optuna 自动搜索 LightGBM 最优超参数.

    搜索空间包括:
      - learning_rate, num_leaves, max_depth
      - min_child_samples, subsample, colsample_bytree
      - reg_alpha, reg_lambda

    Parameters
    ----------
    X_train, y_train : 训练数据.
    X_valid, y_valid : 验证数据.
    n_trials : int
        Optuna 试验次数.
    timeout : int
        搜索超时时间(秒).
    random_state : int
        随机种子.

    Returns
    -------
    best_params : dict
        最优超参数字典.
    """
    def objective(trial: optuna.Trial) -> float:
        """Optuna 目标函数: 最大化验证集 AUC."""

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float(
                "learning_rate",
                LIGHTGBM_PARAM_SPACE["learning_rate"][0],
                LIGHTGBM_PARAM_SPACE["learning_rate"][1],
                log=True,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                LIGHTGBM_PARAM_SPACE["num_leaves"][0],
                LIGHTGBM_PARAM_SPACE["num_leaves"][1],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                LIGHTGBM_PARAM_SPACE["max_depth"][0],
                LIGHTGBM_PARAM_SPACE["max_depth"][1],
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                LIGHTGBM_PARAM_SPACE["min_child_samples"][0],
                LIGHTGBM_PARAM_SPACE["min_child_samples"][1],
            ),
            "subsample": trial.suggest_float(
                "subsample",
                LIGHTGBM_PARAM_SPACE["subsample"][0],
                LIGHTGBM_PARAM_SPACE["subsample"][1],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                LIGHTGBM_PARAM_SPACE["colsample_bytree"][0],
                LIGHTGBM_PARAM_SPACE["colsample_bytree"][1],
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                LIGHTGBM_PARAM_SPACE["reg_alpha"][0],
                LIGHTGBM_PARAM_SPACE["reg_alpha"][1],
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                LIGHTGBM_PARAM_SPACE["reg_lambda"][0],
                LIGHTGBM_PARAM_SPACE["reg_lambda"][1],
                log=True,
            ),
            "scale_pos_weight": 100,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }

        # 用少量 boost rounds 快速评估
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

        model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=500,  # 搜索时只用 500 轮加速
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
            ],
        )

        # 返回验证集最佳 AUC
        auc = model.best_score["valid"]["auc"]
        return auc

    # 创建 Optuna Study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_warmup_steps=10),
    )

    print(f"\n[Optuna] 开始超参数搜索 ({n_trials} trials, timeout={timeout}s)")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    print(f"\n[Optuna] 搜索完成!")
    print(f"  最佳验证 AUC: {study.best_value:.6f}")
    print(f"  最佳超参数: {study.best_params}")

    return study.best_params
