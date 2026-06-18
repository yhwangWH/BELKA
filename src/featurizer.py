"""
特征工程模块
=============

负责将 SMILES 字符串转换为机器学习模型可用的数值特征:

  1. Morgan/ECFP4 圆形指纹 (2048 位)
  2. MACCS Keys 指纹 (167 位, 可选)
  3. 分子理化性质 (MW, LogP, TPSA 等)
  4. 蛋白质 one-hot 编码

所有特征拼接后作为模型输入.
"""

import gc
import json
import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem import MACCSkeys
from rdkit import RDLogger

# 禁用 RDKit 警告输出 (SMILES 解析失败时会产生大量日志)
RDLogger.logger().setLevel(RDLogger.ERROR)

from .config import (
    MORGAN_RADIUS,
    MORGAN_NBITS,
    MACCS_NBITS,
    PHYSICOCHEM_FEATURES,
    PROTEIN_NAMES,
    PROTEIN_TO_IDX,
    NUM_PROTEINS,
    USE_PHYSICOCHEMICAL,
    FEATURE_CHUNK_SIZE,
    FEATURE_MEMMAP_THRESHOLD,
    OUTPUT_DIR,
)


class MolecularFeaturizer:
    """
    分子特征提取器.

    负责:
      - 将 SMILES 字符串解析为 RDKit Mol 对象
      - 计算 Morgan 指纹 / MACCS 指纹
      - 计算理化性质描述符
      - 缓存已解析的分子以避免重复计算 (有上限)

    Usage
    -----
    >>> featurizer = MolecularFeaturizer()
    >>> X = featurizer.fit_transform(df, protein_encoding="onehot")
    """

    # 分子缓存硬上限 (Mol 对象每个约 1-5 KB, 10万限制 ≈ 500 MB 上限)
    _MAX_CACHE_SIZE = 100_000

    def __init__(
        self,
        fingerprint_type: str = "morgan",
        morgan_radius: int = MORGAN_RADIUS,
        morgan_nbits: int = MORGAN_NBITS,
        use_physicochemical: bool = USE_PHYSICOCHEMICAL,
        cache_mols: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        fingerprint_type : str
            指纹类型: "morgan" (默认), "maccs", "both".
        morgan_radius : int
            Morgan 指纹半径 (2 = ECFP4).
        morgan_nbits : int
            Morgan 指纹位数量 (默认 2048).
        use_physicochemical : bool
            是否计算理化性质特征.
        cache_mols : bool
            是否缓存 RDKit Mol 对象以加速后续特征计算.
        """
        self.fingerprint_type = fingerprint_type
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.use_physicochemical = use_physicochemical
        self.cache_mols = cache_mols

        # 分子缓存: {smiles: RDKit Mol} (有大小上限)
        self._mol_cache: Dict[str, Optional[Chem.Mol]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # 特征维度信息 (在第一次 fit_transform 后填充)
        self.feature_names_: List[str] = []
        self.n_features_: int = 0

    # -----------------------------------------------------------------
    # 公共接口
    # -----------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        protein_encoding: str = "onehot",
        smiles_col: str = "molecule_smiles",
        protein_col: str = "protein_name",
    ) -> np.ndarray:
        """
        一站式特征提取: SMILES → 分子指纹 + 理化性质 + 蛋白质编码.

        小数据集直接内存处理; 大数据集自动切块写入 memmap 避免 OOM.

        Parameters
        ----------
        df : pd.DataFrame
            包含 SMILES 列和蛋白质名称列的 DataFrame.
        protein_encoding : str
            蛋白质编码方式: "onehot" (one-hot) 或 "label" (整数标签).
        smiles_col : str
            SMILES 列名.
        protein_col : str
            蛋白质名称列名.

        Returns
        -------
        X : np.ndarray or np.memmap
            特征矩阵 (float32). 大数据集返回 memmap (磁盘-backed).
        """
        n_samples = len(df)

        # 小数据集 → 直接内存处理 (原路径)
        if n_samples <= FEATURE_MEMMAP_THRESHOLD:
            return self._fit_transform_in_memory(df, protein_encoding,
                                                 smiles_col, protein_col)

        # 大数据集 → 分块写入 memmap
        print(f"  [Featurizer] 检测到大数据集 ({n_samples:,} 行), "
              f"启用分块 memmap 模式 (chunk={FEATURE_CHUNK_SIZE:,})")
        return self._fit_transform_to_memmap(df, protein_encoding,
                                             smiles_col, protein_col,
                                             chunk_size=FEATURE_CHUNK_SIZE)

    def _fit_transform_in_memory(
        self,
        df: pd.DataFrame,
        protein_encoding: str,
        smiles_col: str,
        protein_col: str,
    ) -> np.ndarray:
        """小数据集的原地 (in-memory) 特征提取."""
        # 1. 提取分子指纹
        print(f"  [Featurizer] 步骤1/3: 计算分子指纹 ({len(df):,} 个分子)...")
        fps = self._compute_fingerprints(df, smiles_col)

        # 2. 可选: 提取理化性质
        if self.use_physicochemical:
            print(f"  [Featurizer] 步骤2/3: 计算理化性质...")
            physchem = self._compute_physicochemical(df, smiles_col)
        else:
            physchem = None

        # 3. 蛋白质编码
        print(f"  [Featurizer] 步骤3/3: 蛋白质编码...")
        prot_enc = self._encode_proteins(df, protein_col, protein_encoding)

        # 4. 拼接所有特征
        X = self._concatenate_features(fps, physchem, prot_enc)

        # 5. 保存特征名供后续分析
        self._build_feature_names()

        self.n_features_ = X.shape[1]
        print(f"[Featurizer] 特征矩阵: {X.shape[0]:,} × {X.shape[1]} 维 "
              f"({X.nbytes / (1024**2):.1f} MB, dtype={X.dtype})")

        return X

    def _fit_transform_to_memmap(
        self,
        df: pd.DataFrame,
        protein_encoding: str,
        smiles_col: str,
        protein_col: str,
        chunk_size: int = 500_000,
        memmap_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        分块提取特征并写入磁盘 memmap (支持断点续跑).

        工作流:
          1. 计算总特征维度, 预分配 memmap 文件
          2. 检测 checkpoint: 如存在则跳过已完成块, 从断点续跑
          3. 逐块计算特征并写入 memmap
          4. 每完成一块即更新 checkpoint 文件
          5. 全部完成后删除 checkpoint, 返回 memmap

        中断后重新运行: 自动跳过已写入的块, 无需从头计算.
        """
        n_samples = len(df)

        # ---- 计算特征维度 ----
        n_fp = self._get_fp_nbits()
        n_physchem = len(PHYSICOCHEM_FEATURES) if self.use_physicochemical else 0
        n_prot = NUM_PROTEINS if protein_encoding == "onehot" else 1
        n_features = n_fp + n_physchem + n_prot

        # ---- 确定文件路径 ----
        if memmap_path is None:
            memmap_path = os.path.join(OUTPUT_DIR, "features_train.dat")
        checkpoint_path = memmap_path + ".checkpoint"

        # ---- 检测断点续跑 ----
        completed_chunks: set = set()
        is_resume = False

        if os.path.exists(checkpoint_path) and os.path.exists(memmap_path):
            try:
                with open(checkpoint_path, "r") as f:
                    ck = json.load(f)
                # 验证兼容性: 样本数和维度必须匹配
                if (ck.get("n_samples") == n_samples
                        and ck.get("n_features") == n_features
                        and ck.get("chunk_size") == chunk_size):
                    completed_chunks = set(ck.get("completed", []))
                    if completed_chunks:
                        is_resume = True
                        print(f"  [Featurizer] 🔄 检测到断点: {len(completed_chunks)} 个块已完成, "
                              f"从第 {max(completed_chunks) + 2} 块续跑")
                else:
                    print(f"  [Featurizer] ⚠ 旧 checkpoint 不兼容, 重新从头开始")
                    completed_chunks = set()
            except (json.JSONDecodeError, KeyError):
                print(f"  [Featurizer] ⚠ checkpoint 文件损坏, 重新从头开始")
                completed_chunks = set()

        if not is_resume:
            print(f"  [Featurizer] 创建 memmap: {memmap_path}")
            print(f"  [Featurizer] 总特征维度: {n_features} "
                  f"({n_fp} fp + {n_physchem} physchem + {n_prot} prot)")
            estimated_size = n_samples * n_features * 4
            print(f"  [Featurizer] 预估磁盘占用: {estimated_size / (1024**3):.1f} GiB")

        # ---- 创建/打开 memmap ----
        if is_resume:
            X_memmap = np.memmap(
                memmap_path, dtype="float32", mode="r+",
                shape=(n_samples, n_features),
            )
        else:
            X_memmap = np.memmap(
                memmap_path, dtype="float32", mode="w+",
                shape=(n_samples, n_features),
            )

        # ---- 分块处理 ----
        n_chunks = (n_samples + chunk_size - 1) // chunk_size

        for chunk_i in range(n_chunks):
            if chunk_i in completed_chunks:
                continue  # 断点续跑: 跳过已完成块

            start = chunk_i * chunk_size
            end = min(start + chunk_size, n_samples)
            chunk_n = end - start
            print(f"  [Featurizer] ── Chunk {chunk_i + 1}/{n_chunks} "
                  f"({start:,} → {end:,}, {chunk_n:,} 行) ──")

            chunk_df = df.iloc[start:end]

            # -- 指纹 (uint8) --
            fps_u8 = self._compute_fingerprints(chunk_df, smiles_col)

            # -- 理化性质 --
            if self.use_physicochemical:
                physchem = self._compute_physicochemical(chunk_df, smiles_col)
            else:
                physchem = None

            # -- 蛋白质编码 --
            prot_enc = self._encode_proteins(chunk_df, protein_col, protein_encoding)

            # -- 写入 memmap (分批次避免 OOM) --
            write_batch = 10_000
            col = 0

            # 指纹：分批写入，避免 astype(np.float32) 产生过大临时数组
            for b_start in range(0, chunk_n, write_batch):
                b_end = min(b_start + write_batch, chunk_n)
                X_memmap[start + b_start:start + b_end, col:col + n_fp] = \
                    fps_u8[b_start:b_end].astype(np.float32)
            col += n_fp

            # 理化性质：同样分批
            if physchem is not None:
                for b_start in range(0, chunk_n, write_batch):
                    b_end = min(b_start + write_batch, chunk_n)
                    X_memmap[start + b_start:start + b_end, col:col + n_physchem] = \
                        physchem[b_start:b_end]
                col += n_physchem

            X_memmap[start:end, col:col + n_prot] = prot_enc

            # 释放当前块
            del fps_u8, physchem, prot_enc
            gc.collect()

            # ---- 保存 checkpoint (每完成一块立即持久化) ----
            completed_chunks.add(chunk_i)
            self._save_checkpoint(checkpoint_path, completed_chunks,
                                  n_samples, n_features, chunk_size)

        X_memmap.flush()
        self._memmap_path = memmap_path

        # ---- 全部完成, 删除 checkpoint ----
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        print(f"  [Featurizer] ✓ 全部 {n_chunks} 块完成! "
              f"特征矩阵: {n_samples:,} × {n_features} (memmap @ {memmap_path})")

        # 构建特征名
        self._build_feature_names()
        self.n_features_ = n_features

        return X_memmap

    @staticmethod
    def _save_checkpoint(
        path: str,
        completed: set,
        n_samples: int,
        n_features: int,
        chunk_size: int,
    ) -> None:
        """将已完成的块索引原子写入 checkpoint 文件."""
        tmp_path = path + ".tmp"
        data = {
            "completed": sorted(completed),
            "n_samples": n_samples,
            "n_features": n_features,
            "chunk_size": chunk_size,
        }
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # 原子替换, 避免写入中途崩溃导致文件损坏

    def _get_fp_nbits(self) -> int:
        """返回当前配置下的指纹位数."""
        if self.fingerprint_type == "morgan":
            return self.morgan_nbits
        elif self.fingerprint_type == "maccs":
            return MACCS_NBITS
        elif self.fingerprint_type == "both":
            return self.morgan_nbits + MACCS_NBITS
        else:
            raise ValueError(f"不支持的指纹类型: {self.fingerprint_type}")

    def cleanup_memmap(self) -> None:
        """删除 memmap 文件和 checkpoint 以释放磁盘空间."""
        if hasattr(self, "_memmap_path"):
            # 同时清理 checkpoint (可能因中断而残留)
            ck_path = self._memmap_path + ".checkpoint"
            if os.path.exists(ck_path):
                try:
                    os.remove(ck_path)
                except OSError:
                    pass

            if os.path.exists(self._memmap_path):
                try:
                    os.remove(self._memmap_path)
                    print(f"[Featurizer] 已清理 memmap: {self._memmap_path}")
                except OSError as e:
                    print(f"[Featurizer] ⚠ 清理 memmap 失败: {e}")

    def transform_test(
        self,
        df: pd.DataFrame,
        protein_encoding: str = "onehot",
        smiles_col: str = "molecule_smiles",
        protein_col: str = "protein_name",
    ) -> np.ndarray:
        """
        对测试数据提取特征 (与 fit_transform 一致的流程, 但不重新构建特征名).

        Parameters
        ----------
        df : pd.DataFrame
            测试数据 DataFrame.
        protein_encoding : str
            蛋白质编码方式.
        smiles_col : str
            SMILES 列名.
        protein_col : str
            蛋白质名称列名.

        Returns
        -------
        X : np.ndarray
            测试特征矩阵.
        """
        # 复用 fit_transform 逻辑
        return self.fit_transform(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            df,
            protein_encoding=protein_encoding,
            smiles_col=smiles_col,
            protein_col=protein_col,
        )

    def clear_cache(self) -> None:
        """清除分子缓存以释放内存."""
        cache_size = len(self._mol_cache)
        self._mol_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        gc.collect()
        if cache_size > 0:
            print(f"  [Featurizer] 已清除 {cache_size:,} 个分子缓存")

    # -----------------------------------------------------------------
    # 内部: SMILES → RDKit Mol
    # -----------------------------------------------------------------

    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        将 SMILES 字符串解析为 RDKit Mol 对象 (带缓存上限).

        Parameters
        ----------
        smiles : str
            SMILES 字符串.

        Returns
        -------
        mol : rdkit.Chem.Mol or None
            解析成功返回 Mol 对象, 失败返回 None.
        """
        if smiles in self._mol_cache:
            self._cache_hits += 1
            return self._mol_cache[smiles]

        self._cache_misses += 1

        if not isinstance(smiles, str) or not smiles:
            mol = None
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)

        # 缓存上限控制: 超过上限则清空一半
        if self.cache_mols and len(self._mol_cache) >= self._MAX_CACHE_SIZE:
            items = list(self._mol_cache.items())
            self._mol_cache = dict(items[len(items) // 2:])

        if self.cache_mols:
            self._mol_cache[smiles] = mol

        return mol

    # -----------------------------------------------------------------
    # 内部: 指纹计算
    # -----------------------------------------------------------------

    def _compute_fingerprints(
        self, df: pd.DataFrame, smiles_col: str
    ) -> np.ndarray:
        """
        批量计算分子指纹.

        Parameters
        ----------
        df : pd.DataFrame
            数据.
        smiles_col : str
            SMILES 列名.

        Returns
        -------
        fps : np.ndarray, shape (n_samples, n_fp_bits)
            分子指纹矩阵.
        """
        n_samples = len(df)
        smiles_list = df[smiles_col].values

        # 根据指纹类型分配存储数组
        if self.fingerprint_type == "morgan":
            n_bits = self.morgan_nbits
        elif self.fingerprint_type == "maccs":
            n_bits = MACCS_NBITS
        elif self.fingerprint_type == "both":
            n_bits = self.morgan_nbits + MACCS_NBITS
        else:
            raise ValueError(f"不支持的指纹类型: {self.fingerprint_type}")

        fps = np.zeros((n_samples, n_bits), dtype=np.uint8)
        n_errors = 0

        for i in range(n_samples):
            smiles = str(smiles_list[i])
            mol = self._smiles_to_mol(smiles)

            if mol is None:
                n_errors += 1
                continue

            if self.fingerprint_type == "morgan":
                fp = self._get_morgan_fp(mol)
                fps[i, :] = fp
            elif self.fingerprint_type == "maccs":
                fp = self._get_maccs_fp(mol)
                fps[i, :] = fp
            elif self.fingerprint_type == "both":
                fp_m = self._get_morgan_fp(mol)
                fp_maccs = self._get_maccs_fp(mol)
                fps[i, : self.morgan_nbits] = fp_m
                fps[i, self.morgan_nbits:] = fp_maccs

            # 每 50000 条打印一次进度
            if (i + 1) % 50000 == 0:
                pct = 100 * (i + 1) / n_samples
                print(f"  [Featurizer] 指纹进度: {i+1:,}/{n_samples:,} ({pct:.1f}%)")

        if n_errors > 0:
            print(f"  [Featurizer] ⚠ 警告: {n_errors} 个 SMILES 解析失败 "
                  f"({100 * n_errors / n_samples:.2f}%)")

        # 打印缓存统计
        if self._cache_hits + self._cache_misses > 0:
            hit_rate = 100 * self._cache_hits / (self._cache_hits + self._cache_misses)
            print(f"  [Featurizer] Mol 缓存: {len(self._mol_cache):,} 条目, "
                  f"命中率 {hit_rate:.1f}%")

        # 释放 Mol 缓存
        self.clear_cache()

        return fps

    def _get_morgan_fp(self, mol: Chem.Mol) -> np.ndarray:
        """
        计算单个分子的 Morgan/ECFP 指纹.

        使用 AllChem.GetMorganFingerprintAsBitVect 获取位向量,
        转换为 numpy 数组.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            已解析的分子 (含氢).

        Returns
        -------
        fp_arr : np.ndarray, shape (nBits,)
            Morgan 指纹位向量 (0/1).
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            self.morgan_radius,
            nBits=self.morgan_nbits,
            useChirality=True,     # 考虑手性信息
        )
        arr = np.zeros(self.morgan_nbits, dtype=np.uint8)
        # 将 RDKit BitVect 转换为 numpy 数组
        # 使用 DataStructs 转换效率更高
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        return arr

    def _get_maccs_fp(self, mol: Chem.Mol) -> np.ndarray:
        """
        计算单个分子的 MACCS Keys 指纹 (167 位).

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            已解析的分子 (含氢).

        Returns
        -------
        fp_arr : np.ndarray, shape (167,)
            MACCS Keys 指纹位向量 (0/1).
        """
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(MACCS_NBITS, dtype=np.uint8)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        return arr

    # -----------------------------------------------------------------
    # 内部: 理化性质
    # -----------------------------------------------------------------

    def _compute_physicochemical(
        self, df: pd.DataFrame, smiles_col: str
    ) -> np.ndarray:
        """
        批量计算分子的理化性质描述符.

        计算的性质包括:
          - MolWt: 分子量
          - LogP: 脂水分配系数 (Crippen)
          - NumHAcceptors: 氢键受体数
          - NumHDonors: 氢键供体数
          - NumRotatableBonds: 可旋转键数
          - TPSA: 拓扑极性表面积
          - FractionCsp3: sp3 碳比例
          - NumAromaticRings: 芳环数
          - NumSaturatedRings: 饱和环数
          - NumAliphaticRings: 脂肪环数

        Parameters
        ----------
        df : pd.DataFrame
            数据.
        smiles_col : str
            SMILES 列名.

        Returns
        -------
        physchem : np.ndarray, shape (n_samples, n_properties)
            理化性质矩阵 (float32).
        """
        # 计算函数列表 (按 PHYSICOCHEM_FEATURES 顺序, 直接引用避免 lambda 间接调用)
        _PROPERTY_FNS = [
            Descriptors.MolWt,
            Crippen.MolLogP,
            Lipinski.NumHAcceptors,
            Lipinski.NumHDonors,
            Lipinski.NumRotatableBonds,
            rdMolDescriptors.CalcTPSA,
            rdMolDescriptors.CalcFractionCSP3,
            rdMolDescriptors.CalcNumAromaticRings,
            rdMolDescriptors.CalcNumSaturatedRings,
            rdMolDescriptors.CalcNumAliphaticRings,
        ]

        n_samples = len(df)
        n_props = len(_PROPERTY_FNS)
        physchem = np.zeros((n_samples, n_props), dtype=np.float32)

        report_interval = max(50000, n_samples // 20)  # 至少每5%打印一次

        for i in range(n_samples):
            smiles = str(df[smiles_col].iloc[i])
            mol = self._smiles_to_mol(smiles)

            if mol is not None:
                for j, fn in enumerate(_PROPERTY_FNS):
                    try:
                        val = fn(mol)
                        physchem[i, j] = float(val) if val is not None else 0.0
                    except Exception:
                        pass  # 某些描述符可能在特定分子上计算失败, 填 0

            # 进度输出
            if (i + 1) % report_interval == 0:
                pct = 100 * (i + 1) / n_samples
                print(f"  [Featurizer] 理化性质: {i+1:,}/{n_samples:,} ({pct:.1f}%)")

        return physchem

    # -----------------------------------------------------------------
    # 内部: 蛋白质编码
    # -----------------------------------------------------------------

    def _encode_proteins(
        self,
        df: pd.DataFrame,
        protein_col: str,
        encoding: str,
    ) -> np.ndarray:
        """
        对蛋白质名称进行编码.

        Parameters
        ----------
        df : pd.DataFrame
            数据.
        protein_col : str
            蛋白质名称列名.
        encoding : str
            "onehot" → one-hot (3维), "label" → 整数标签 (1维).

        Returns
        -------
        encoded : np.ndarray
            编码后的蛋白质特征.
        """
        proteins = df[protein_col].values

        if encoding == "onehot":
            # 向量化: 用 pd.Categorical 直接映射为整数索引
            cat = pd.Categorical(
                [str(p) for p in proteins],
                categories=PROTEIN_NAMES,
            )
            encoded = np.zeros((len(proteins), NUM_PROTEINS), dtype=np.float32)
            valid = cat.codes >= 0
            encoded[valid, cat.codes[valid]] = 1.0
        elif encoding == "label":
            # 向量化 one-hot 的子集转为单列标签
            cat = pd.Categorical(
                [str(p) for p in proteins],
                categories=PROTEIN_NAMES,
            )
            encoded = cat.codes.astype(np.float32).reshape(-1, 1)
        else:
            raise ValueError(f"不支持的蛋白质编码方式: {encoding}")

        return encoded

    # -----------------------------------------------------------------
    # 内部: 特征拼接
    # -----------------------------------------------------------------

    def _concatenate_features(
        self,
        fps: np.ndarray,
        physchem: Optional[np.ndarray],
        prot_enc: np.ndarray,
    ) -> np.ndarray:
        """
        将指纹、理化性质和蛋白质编码拼接为统一特征矩阵.

        Parameters
        ----------
        fps : np.ndarray
            分子指纹.
        physchem : np.ndarray or None
            理化性质.
        prot_enc : np.ndarray
            蛋白质编码.

        Returns
        -------
        X : np.ndarray
            拼接后的特征矩阵.
        """
        parts = [fps]

        if physchem is not None:
            parts.append(physchem)

        parts.append(prot_enc)

        X = np.concatenate(parts, axis=1).astype(np.float32)
        return X

    def _build_feature_names(self) -> None:
        """构造特征名列表, 用于后续特征重要性分析."""
        names = []

        # 指纹特征名
        if self.fingerprint_type == "morgan":
            names += [f"morgan_{i}" for i in range(self.morgan_nbits)]
        elif self.fingerprint_type == "maccs":
            names += [f"maccs_{i}" for i in range(MACCS_NBITS)]
        elif self.fingerprint_type == "both":
            names += [f"morgan_{i}" for i in range(self.morgan_nbits)]
            names += [f"maccs_{i}" for i in range(MACCS_NBITS)]

        # 理化性质特征名
        if self.use_physicochemical:
            names += PHYSICOCHEM_FEATURES

        # 蛋白质编码特征名
        names += [f"protein_{p}" for p in ["BRD4", "HSA", "sEH"]]

        self.feature_names_ = names
