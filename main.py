"""
BELKA Kaggle Competition - Phase 1 Baseline
============================================

主入口脚本, 运行端到端训练流程.

快速验证模式 (推荐先用这个):
    python main.py

全量训练模式:·
    python main.py --full

启用 Optuna 超参数搜索:
    python main.py --full --optuna

输出:
    - output/models/        : 训练好的 LightGBM 模型 (每个 fold 一个)
    - output/submissions/   : Kaggle 提交文件
    - 终端输出              : 交叉验证分数汇总
"""

import argparse
import os
import sys
import time
from datetime import datetime

# --- Kaggle 环境路径清理 ---
# 在 Kaggle 上, 当前文件可能加载自 /kaggle/input/datasets/*/belka-kaggle3/,
# 但 sys.path 中可能有旧的 belka 相关路径和缓存的 src 模块导致导入失败.
# 清理旧路径和缓存后重新导入.
_IS_KAGGLE = os.path.exists("/kaggle/working")
if _IS_KAGGLE:
    # 清理已加载的 belka/src 模块缓存
    for key in list(sys.modules.keys()):
        if any(kw in key.lower() for kw in ("src", "belka")):
            del sys.modules[key]
    # 清理旧路径
    sys.path = [p for p in sys.path if "belka" not in p.lower()]

# 确保项目根目录在 sys.path 中 (本地和 Kaggle 环境均兼容)
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.pipeline import BELKAPipeline


def main():
    t_start = time.time()
    print(f"\n{'='*65}")
    print(f"  BELKA Pipeline 启动")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"{'='*65}")
    print(f"\n  💡 提示: 以下各步骤会输出进度, 如果长时间没有新输出")
    print(f"     请检查终端是否卡死 (可尝试按 Enter 键刷新缓冲区)\n")

    parser = argparse.ArgumentParser(
        description="BELKA Phase 1: LightGBM Baseline Training"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="全量数据训练 (默认快速验证 300 行)",
    )
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="启用 Optuna 超参数自动搜索",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认 42)",
    )
    args = parser.parse_args()
    print(f"  → 参数: --full={args.full}, --optuna={args.optuna}, --seed={args.seed}")

    # 初始化 pipeline
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] 初始化 Pipeline...")
    pipeline = BELKAPipeline(
        quick_mode=not args.full,
        use_optuna=args.optuna,
        random_seed=args.seed,
    )

    # 执行完整流程
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] 开始执行流程 (进度见下方)...\n")
    results = pipeline.run()

    # 打印最终结果
    elapsed = time.time() - t_start
    if elapsed < 60:
        time_str = f"{elapsed:.1f} 秒"
    else:
        time_str = f"{elapsed / 60:.1f} 分钟"
    print(f"\n{'='*65}")
    print(f"  Pipeline 执行完毕!")
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总耗时:   {time_str}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
