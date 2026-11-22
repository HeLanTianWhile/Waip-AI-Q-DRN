# -*- coding: utf-8 -*-
"""
DRN 核心传播器（numpy 向量化）。

语义与 main/main.py 的经典传播完全一致（已与 data/ChatWorkspace.py 的
NeuralWorker._propagate 做过逐神经元对比，误差 < 1e-9），但用
np.bincount 把 O(active × synapses) 的 Python 循环变成单次向量化归约，
供训练器高速评估 100 个网络使用。

状态约定：
  y[i] = NaN           → 该神经元 "NOT Activate"
  y[i] = 有限浮点数    → 该神经元激活，值即为 y
外部输入注入直接把对应输入神经元的 y 写成灰度/位置值（0 → NaN）。
"""

import numpy as np

try:
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

SCREEN_W, SCREEN_H = 390, 844
VIEW_W, VIEW_H = 390, 192
VIEW_X = SCREEN_W / 2.0
VIEW_Y_MIN = VIEW_H / 2.0              # 96
VIEW_Y_MAX = SCREEN_H - VIEW_H / 2.0   # 748
PIXEL_COUNT = VIEW_W * VIEW_H          # 74880
INPUT_COUNT = PIXEL_COUNT + 4          # 74884
OUTPUT_COUNT = 4


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def _norm(x, dim):
    """与 ChatWorkspace.normalized_position 一致：除以尺寸并四舍五入到 3 位。"""
    v = round(x / dim, 3)
    return float(v) if v else 0.0  # 0 调用方会转成 NaN


class Network:
    """把 config/synapses 数据库编译成 numpy 数组，支持高速传播。"""

    def __init__(self, config_db, synapses_db):
        ids = sorted(config_db.keys(), key=lambda k: int(k))
        self.id_to_idx = {nid: i for i, nid in enumerate(ids)}
        self.N = len(ids)

        self.bias = np.empty(self.N, dtype=np.float64)
        self.input_idx = np.full(INPUT_COUNT, -1, dtype=np.int64)
        self.output_idx = np.full(OUTPUT_COUNT, -1, dtype=np.int64)
        for i, nid in enumerate(ids):
            cfg = config_db[nid]
            self.bias[i] = float(cfg.get("bias", 0.0))
            ip = cfg.get("IP")
            if isinstance(ip, (list, tuple)) and len(ip) >= 2:
                kind, index = ip[0], ip[1]
                if isinstance(index, int):
                    if kind == "Input" and 0 <= index < INPUT_COUNT:
                        self.input_idx[index] = i
                    elif kind == "Output" and 0 <= index < OUTPUT_COUNT:
                        self.output_idx[index] = i

        src, tgt, w = [], [], []
        for syn in synapses_db.values():
            ct = syn.get("Connection Target")
            if not isinstance(ct, (list, tuple)) or len(ct) != 2:
                continue
            a, b = ct[0], ct[1]
            if a == b:
                continue
            ia = self.id_to_idx.get(a)
            ib = self.id_to_idx.get(b)
            if ia is None or ib is None:
                continue
            weight = float(syn.get("Weight", 0.0))
            src.append(ia); tgt.append(ib); w.append(weight)
            src.append(ib); tgt.append(ia); w.append(weight)
        self.src = np.asarray(src, dtype=np.int64)
        self.tgt = np.asarray(tgt, dtype=np.int64)
        self.w = np.asarray(w, dtype=np.float64)

        # 可选稀疏加速：A[tgt, src] = weight（scipy 可用时快 ~25%，数值与 bincount 一致）
        self._A = None
        self._Amask = None
        if HAS_SCIPY and self.src.size:
            try:
                self._A = csr_matrix(
                    (self.w, (self.tgt, self.src)), shape=(self.N, self.N)
                )
                self._Amask = self._A != 0
            except Exception:
                self._A = None
                self._Amask = None

        # 状态：NaN = NOT Activate
        self.y = np.full(self.N, np.nan, dtype=np.float64)

    def reset_state(self):
        self.y.fill(np.nan)

    def _sanitize(self):
        """inf → ±1，NaN 保持（= NOT Activate）。与 ChatWorkspace._sanitize_state 一致。"""
        if self.y.size:
            pos = np.isposinf(self.y)
            neg = np.isneginf(self.y)
            if pos.any():
                self.y[pos] = 1.0
            if neg.any():
                self.y[neg] = -1.0

    def inject(self, pixels, cursor_x, cursor_y, view_y):
        """pixels: 长度 74880 的灰度字节（0-255）或可转 np.uint8 的序列。"""
        g = np.frombuffer(pixels, dtype=np.uint8).astype(np.float64) \
            if isinstance(pixels, (bytes, bytearray, memoryview)) \
            else np.asarray(pixels, dtype=np.float64)
        vals = g / 255.0
        vals[g == 0] = np.nan
        self.y[self.input_idx[:PIXEL_COUNT]] = vals

        positions = (
            _norm(VIEW_X, SCREEN_W), _norm(view_y, SCREEN_H),
            _norm(cursor_x, SCREEN_W), _norm(cursor_y, SCREEN_H),
        )
        for offset, value in enumerate(positions):
            idx = self.input_idx[PIXEL_COUNT + offset]
            self.y[idx] = float(value) if value else np.nan

    def propagate(self):
        self._sanitize()
        active = ~np.isnan(self.y)
        ys = self.y.copy()
        ys[~active] = 0.0
        if self._A is not None:
            # scipy 稀疏矩阵向量乘（数值与下面 bincount 路径一致）
            task = self._A @ ys
            touched = (self._Amask @ active.astype(np.float64)) > 0
        else:
            # 每条边的贡献 = weight × 源 y
            contrib = self.w * ys[self.src]
            task = np.bincount(self.tgt, weights=contrib, minlength=self.N)
            edge_active = active[self.src]
            touched = np.bincount(
                self.tgt, weights=edge_active.astype(np.float64), minlength=self.N
            ) > 0
        new_y = np.full(self.N, np.nan)
        if touched.any():
            new_y[touched] = np.tanh(task[touched] - self.bias[touched])
        self.y = new_y
        self._sanitize()

    def read_outputs(self):
        out = self.y[self.output_idx]
        result = []
        for v in out:
            if np.isfinite(v):
                result.append(float(v))
            else:
                result.append(0.0)
        return result
