"""
全局配置常量
=================

集中管理所有硬编码路径、超参数和特征工程参数。
方便后续实验修改和一键切换配置。
"""

import os

# ============================================================================
# 路径配置
# ============================================================================
# 自动检测是否在 Kaggle 环境中 (只读 /kaggle/input, 可写 /kaggle/working)
_IS_KAGGLE = os.path.exists("/kaggle/working")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# 输出目录: Kaggle 环境必须指向 /kaggle/working (其他路径只读)
if _IS_KAGGLE:
    OUTPUT_DIR = "/kaggle/working"
else:
    OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submissions")

# NOTE: 不在 import 阶段创建目录, 避免 Kaggle 只读文件系统报错.
#       目录将在 pipeline 中首次需要写文件时由 ensure_output_dirs() 创建.


def ensure_output_dirs() -> None:
    """创建输出子目录 (在需要写文件时调用, 而非 import 时)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


# --- Kaggle 竞赛数据路径自动检测 ---
# 竞赛数据总是挂载在 /kaggle/input/ 下的某个竞赛目录中.
# 优先使用竞赛官方数据, 其次使用项目自身 data/ 目录.
_KAGGLE_INPUT_DIR = "/kaggle/input"

def _find_competition_data() -> tuple:
    """
    在 Kaggle 环境中自动定位竞赛数据文件.

    遍历 /kaggle/input/ 下所有子目录, 查找 train.parquet / test.parquet /
    sample_submission.csv. 找到包含这些文件的目录即视为竞赛数据目录.

    Returns
    -------
    (train_path, test_path, sample_sub_path) : tuple of str
        三个数据文件的绝对路径.
    """
    if not _IS_KAGGLE:
        return (
            os.path.join(DATA_DIR, "train.parquet"),
            os.path.join(DATA_DIR, "test.parquet"),
            os.path.join(DATA_DIR, "sample_submission.csv"),
        )

    train_path = test_path = sample_path = ""

    # 遍历所有 /kaggle/input/ 子目录
    for dirpath, _, filenames in os.walk(_KAGGLE_INPUT_DIR):
        if dirpath == _KAGGLE_INPUT_DIR:
            continue  # 跳过根目录, 只看子目录
        lower_filenames = [f.lower() for f in filenames]
        for f in filenames:
            f_lower = f.lower()
            if "train.parquet" in f_lower and not train_path:
                train_path = os.path.join(dirpath, f)
            elif "test.parquet" in f_lower and not test_path:
                test_path = os.path.join(dirpath, f)
            elif "sample_submission" in f_lower and not sample_path:
                sample_path = os.path.join(dirpath, f)

        # 找到三个文件就停止
        if train_path and test_path and sample_path:
            break

    # 如果没找到, 回退到项目 data/ 目录
    if not train_path:
        train_path = os.path.join(DATA_DIR, "train.parquet")
    if not test_path:
        test_path = os.path.join(DATA_DIR, "test.parquet")
    if not sample_path:
        sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

    return train_path, test_path, sample_path


TRAIN_FILE, TEST_FILE, SAMPLE_SUB = _find_competition_data()

if _IS_KAGGLE:
    # 在 Kaggle 上打印实际使用的数据路径, 方便调试
    print(f"[config] Kaggle 环境检测到")
    print(f"  TRAIN_FILE  = {TRAIN_FILE}")
    print(f"  TEST_FILE   = {TEST_FILE}")
    print(f"  SAMPLE_SUB  = {SAMPLE_SUB}")
    print(f"  OUTPUT_DIR  = {OUTPUT_DIR}")

# ============================================================================
# 蛋白质靶标配置
# ============================================================================
# 三个蛋白质靶标: EPHX2(sEH), BRD4, ALB(HSA)
PROTEIN_NAMES = ["BRD4", "HSA", "sEH"]

# 蛋白质名称到索引的映射 (用于 one-hot 编码)
PROTEIN_TO_IDX = {name: i for i, name in enumerate(PROTEIN_NAMES)}
NUM_PROTEINS = len(PROTEIN_NAMES)

# ============================================================================
# 特征工程配置
# ============================================================================
# Morgan/ECFP4 指纹参数
FINGERPRINT_TYPE = "morgan"          # 指纹类型: "morgan", "maccs", "both"
MORGAN_RADIUS = 2                    # ECFP4 ~ Morgan radius=2
MORGAN_NBITS = 2048                  # 指纹位数量

# MACCS Keys 指纹 (固定 167 位)
MACCS_NBITS = 167

# 分子理化性质 (额外特征)
USE_PHYSICOCHEMICAL = True           # 是否计算理化性质
PHYSICOCHEM_FEATURES = [
    "MolWt",                          # 分子量
    "LogP",                           # 脂水分配系数
    "NumHAcceptors",                  # 氢键受体数
    "NumHDonors",                     # 氢键供体数
    "NumRotatableBonds",             # 可旋转键数
    "TPSA",                          # 拓扑极性表面积
    "FractionCsp3",                  # sp3 碳比例
    "NumAromaticRings",              # 芳环数
    "NumSaturatedRings",             # 饱和环数
    "NumAliphaticRings",             # 脂肪环数
]

# 蛋白质编码方式: "onehot" 或 "label"
PROTEIN_ENCODING = "onehot"

# 是否对 building blocks 单独生成指纹 (非仅分子指纹)
USE_BUILDING_BLOCK_FEATURES = False   # 第一阶段暂不开, 控制特征维度

# ============================================================================
# 采样配置
# ============================================================================
# 正负样本比例 (负样本数 / 正样本数)
NEGATIVE_RATIO = 20                   # 推荐 10~50

# 全量正样本的最大采样量 (None 表示全部使用)
MAX_POSITIVE_SAMPLES = None           # 如果正样本过多, 可限制上限

# 按 building blocks 抽样以保持化学多样性
USE_CHEMICAL_DIVERSITY_SAMPLING = False  # 第一阶段先不用, 第二阶段实验

# 随机种子
RANDOM_SEED = 42

# ============================================================================
# 模型训练配置
# ============================================================================
# 交叉验证折数
N_FOLDS = 5

# LightGBM 基线参数
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": 8,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "scale_pos_weight": 200,           # 正样本权重放大 (与 is_unbalance 二选一)
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

# 早停参数
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100

# XGBoost 基线参数 (备用)
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 200,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

# ============================================================================
# Optuna 超参数搜索配置
# ============================================================================
OPTUNA_N_TRIALS = 30                   # 搜索试验次数
OPTUNA_TIMEOUT = 3600 * 3              # 搜索超时(秒), 默认3小时

# LightGBM 超参搜索空间
LIGHTGBM_PARAM_SPACE = {
    "learning_rate": (0.01, 0.3),
    "num_leaves": (31, 255),
    "max_depth": (5, 15),
    "min_child_samples": (10, 200),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": (1e-8, 10.0),
    "reg_lambda": (1e-8, 10.0),
}

# ============================================================================
# 分块处理配置 (避免 OOM)
# ============================================================================
# 特征提取时的分块大小 (行数).
# 每块在内存中约需 chunk_size × n_features × 4 bytes.
# 默认 500K → 每块峰值内存约 5 GB.
FEATURE_CHUNK_SIZE = 500_000

# 超过此行数自动切换为分块+memmap模式
FEATURE_MEMMAP_THRESHOLD = 500_000

# ============================================================================
# 内存优化配置
# ============================================================================
# 读取时使用的数据类型 (降低内存占用)
DTYPE_MAP = {
    "id": "int32",
    "buildingblock1_smiles": "category",
    "buildingblock2_smiles": "category",
    "buildingblock3_smiles": "category",
    "molecule_smiles": "category",
    "protein_name": "category",
    "binds": "int8",
}

# 读取列 (按需加载, 避免全部读入内存)
TRAIN_COLUMNS = [
    "buildingblock1_smiles",
    "buildingblock2_smiles",
    "buildingblock3_smiles",
    "molecule_smiles",
    "protein_name",
    "binds",
]

TEST_COLUMNS = [
    "id",
    "buildingblock1_smiles",
    "buildingblock2_smiles",
    "buildingblock3_smiles",
    "molecule_smiles",
    "protein_name",
]
