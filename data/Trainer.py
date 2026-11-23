# -*- coding: utf-8 -*-
"""
遗传算法训练引擎（第一课：精准点击 W 键）。

流程（每一代）：
  1. 100 个网络各跑 50 tick 无头评估 → 得到鼠标轨迹 + 点击位置；
  2. 用 main/score.py 给 100 个网络打分；
  3. 用 main/rank.py 排序，取前 10 为存活网络；
  4. 每个存活网络用 main/mutation.py 生成 9 个子网络（变异率互不相同）；
  5. 10 存活 + 90 子代 = 100，作为下一代继续演化。

持久化：data/TRAIN/
  - screen.npy          静态屏幕缓存
  - train.log           详细训练日志（控制台可查看）
  - generation.json     当前代元数据（供控制台网络管理表）
  - net_000/ ... net_099/   当前代 100 个网络的紧凑存档
  - replays/current/net_*.json   当前代每个网络的 50 tick 回放轨迹
若 TRAIN 已存在则从存档继续，否则用 data_into.py 初始化 100 个网络。
"""

import gc
import importlib.util
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import QObject, Signal

import Simulator
import drn_core

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MAIN_DIR = ROOT_DIR / "main"
TRAIN_DIR = DATA_DIR / "TRAIN"
LOCK_FILE = TRAIN_DIR / "training.lock"

POPULATION = 100
SURVIVORS = 10
CHILDREN_PER_SURVIVOR = 9  # 10 * 9 = 90 子代
TICKS_PER_RUN = 50

# 9 个子网络的变异率互不相同（围绕默认 0.001 几何展开，可按需调整）
MUTATION_RATES = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]

# 自适应变异：前 10 名得分趋同（多样性低）时，大幅提高变异率
CONVERGE_STD = 0.05          # 前10名标准差低于此值 → 按趋同率放大变异率
MUTATION_BOOST_MAX = 10.0    # 完全趋同（σ=0）时的放大倍数
MUTATION_RATE_CAP = 0.32     # 单个子代变异率上限（防止一次改动全部突触）


def adaptive_rates(scores):
    """按前 10 名得分分布返回 (boost, 实际变异率列表, 标准差)。

    前 10 名得分越集中（趋同），变异率放大越多，以跳出局部最优。
    """
    n = len(scores)
    if n == 0:
        return 1.0, list(MUTATION_RATES), 0.0
    mean = sum(scores) / n
    std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5
    if std < CONVERGE_STD:
        # 趋同率越高（σ 越接近 0），放大倍数越大；σ 接近 0.05 时仅略高于 1 倍
        converge_ratio = 1.0 - std / CONVERGE_STD      # 0 ~ 1
        boost = round(1.0 + (MUTATION_BOOST_MAX - 1.0) * converge_ratio, 2)
    else:
        boost = 1.0
    rates = [r * boost for r in MUTATION_RATES]
    if rates and rates[-1] > MUTATION_RATE_CAP:
        # 超出上限时整体按几何比例压缩到上限，保持 9 个子代变异率互不相同
        lo, hi = rates[0], MUTATION_RATE_CAP
        ratio = (hi / lo) ** (1.0 / (len(rates) - 1))
        rates = [lo * ratio ** i for i in range(len(rates))]
    return boost, rates, std


ROLE_CODE = {"Input": 0, "Output": 1, "Memory": 2, "Normal": 3}
CODE_ROLE = {v: k for k, v in ROLE_CODE.items()}


# ---------- 加载 main/ 下的模块（与 Terminal.py 一致，用文件路径加载） ----------

def _load_main_module(name, filename):
    path = MAIN_DIR / filename
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- 网络紧凑存档（比 JSON 小数十倍，读写快） ----------
# 内存中的种群也只保存紧凑数组：(biases, roles, ipidx, a, b, w)。
# 一份完整字典约 180MB，100 份就是十几 GB（旧实现因此 MemoryError）；
# 紧凑数组每份约 6MB，只有在调用 main/mutation.py 时才临时还原字典。

def to_compact(config_db, synapses_db):
    """字典数据库 → (biases, roles, ipidx, a, b, w)。"""
    ids = sorted(config_db.keys(), key=lambda k: int(k))
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    biases = np.empty(len(ids), dtype=np.float64)
    roles = np.empty(len(ids), dtype=np.int8)
    ipidx = np.empty(len(ids), dtype=np.int32)
    for i, nid in enumerate(ids):
        cfg = config_db[nid]
        biases[i] = float(cfg.get("bias", 0.0))
        ip = cfg.get("IP", ["Normal", i])
        roles[i] = ROLE_CODE.get(ip[0], 3)
        ipidx[i] = int(ip[1]) if isinstance(ip[1], int) else i
    a, b, w = [], [], []
    for syn in synapses_db.values():
        ct = syn.get("Connection Target")
        if not isinstance(ct, (list, tuple)) or len(ct) != 2:
            continue
        x, y = ct[0], ct[1]
        if x not in id_to_idx or y not in id_to_idx:
            continue
        a.append(id_to_idx[x]); b.append(id_to_idx[y])
        w.append(float(syn.get("Weight", 0.0)))
    return (biases, roles, ipidx,
            np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64),
            np.asarray(w, dtype=np.float64))


def to_dicts(compact):
    """紧凑数组 → (config_db, synapses_db)，键名规则与旧版一致。"""
    biases, roles, ipidx, a, b, w = compact
    n = biases.size
    config_db, synapses_db = {}, {}
    for i in range(n):
        nid = f"{i + 1:04d}"
        config_db[nid] = {"bias": float(biases[i]),
                          "IP": [CODE_ROLE[int(roles[i])], int(ipidx[i])]}
    for k in range(a.size):
        synapses_db[f"{k + 1:04d}"] = {
            "Weight": float(w[k]),
            "Connection Target": [f"{int(a[k]) + 1:04d}", f"{int(b[k]) + 1:04d}"],
        }
    return config_db, synapses_db


def copy_compact(compact):
    return tuple(arr.copy() for arr in compact)


def save_compact(net_dir, compact):
    net_dir = Path(net_dir)
    net_dir.mkdir(parents=True, exist_ok=True)
    biases, roles, ipidx, a, b, w = compact
    np.save(net_dir / "biases.npy", biases)
    np.save(net_dir / "roles.npy", roles)
    np.save(net_dir / "ipidx.npy", ipidx)
    np.savez(net_dir / "synapses.npz", a=a, b=b, w=w)


def load_compact(net_dir):
    net_dir = Path(net_dir)
    biases = np.load(net_dir / "biases.npy")
    roles = np.load(net_dir / "roles.npy")
    ipidx = np.load(net_dir / "ipidx.npy")
    with np.load(net_dir / "synapses.npz") as syn:
        a, b, w = syn["a"].copy(), syn["b"].copy(), syn["w"].copy()
    return biases, roles, ipidx, a, b, w


def save_network(net_dir, config_db, synapses_db):
    save_compact(net_dir, to_compact(config_db, synapses_db))


def load_network(net_dir):
    return to_dicts(load_compact(net_dir))


# ---------- 训练标记（判断上次训练是否未完成） ----------

def pid_alive(pid):
    """跨平台判断进程是否存活（Windows 用 OpenProcess，避免 os.kill 误杀）。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def write_lock(generation, target_gen=None, state="running"):
    """写入/更新训练标记。异常退出时会残留，用于下次“继续训练”。"""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "state": state,
        "pid": os.getpid(),
        "generation": int(generation),
        "target_gen": None if target_gen is None else int(target_gen),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        LOCK_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    except Exception:
        pass
    return data


def read_lock():
    """读取训练标记；不存在或损坏返回 None。"""
    if not LOCK_FILE.exists():
        return None
    try:
        data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def clear_lock():
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


# ---------- 训练器 ----------


class _Aborted(Exception):
    """用户请求结束训练（任何阶段都可抛出，确保不保存半成品）。"""



class Trainer(QObject):
    sig_log = Signal(str)               # 新日志行
    sig_generation_done = Signal(dict)  # 一代结束的元数据
    sig_progress = Signal(int, int, float)  # (net_index, total, score)
    sig_finished = Signal()             # 训练线程结束（正常完成或被结束）

    def __init__(self, screen, target_gen=None, cursor_sprite=None):
        super().__init__()
        self.screen = screen
        self.cursor_sprite = cursor_sprite  # (灰度, alpha) 仿真光标
        self.target_gen = target_gen    # 绝对值：generation 达到该值即停；None=一直训练
        self._stop = threading.Event()
        self._done = False              # 是否已正常跑完目标代数
        self._thread = None
        self.population = []            # [compact, ...] 紧凑数组，非字典
        self.generation = 0
        self.current_meta = {}          # 最新一代元数据
        self.live_gen = None            # 正在评估的代
        self.live_results = {}          # {idx: {score, clicks, hit_w}} 实时结果
        self._score_fn = _load_main_module("drn_score", "score.py").score
        self._rank_fn = _load_main_module("drn_rank", "rank.py").rank
        self._mutation_fn = _load_main_module("drn_mutation", "mutation.py").mutation
        self._data_into_fn = _load_main_module("drn_data_into", "data_into.py").data_into

    # ---- 生命周期 ----

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._done = False
        write_lock(self.generation, self.target_gen, "running")
        # 让主线程（界面/观看）更容易抢到 GIL
        try:
            sys.setswitchinterval(0.0005)
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._lower_thread_priority(self._thread)

    @staticmethod
    def _lower_thread_priority(thread):
        """Windows：把训练线程降到 IDLE，避免它把界面线程饿死。"""
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenThread(0x0020 | 0x0040, False, thread.native_id)  # SET|QUERY
            if h:
                k32.SetThreadPriority(h, -15)  # THREAD_PRIORITY_IDLE：界面/操作优先
                k32.CloseHandle(h)
        except Exception:
            pass

    def stop(self):
        self._stop.set()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    # ---- 日志 ----

    def _log(self, line=""):
        ts = datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}] {line}" if line else ""
        try:
            with open(TRAIN_DIR / "train.log", "a", encoding="utf-8") as f:
                f.write(full + "\n")
        except Exception:
            pass
        self.sig_log.emit(full)

    # ---- 初始化 / 加载 ----

    def _load_or_init(self):
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        gen_file = TRAIN_DIR / "generation.json"
        first_net = TRAIN_DIR / "net_000" / "biases.npy"
        if gen_file.exists() and first_net.exists():
            try:
                meta = json.loads(gen_file.read_text(encoding="utf-8"))
                self.generation = int(meta.get("generation", 0))
                self._log("=" * 64)
                self._log(f"检测到第 {self.generation} 代存档，正在加载 {POPULATION} 个网络…")
                if self.target_gen is not None:
                    self._log(f"本次目标：跑到第 {self.target_gen} 代"
                              f"（再训练 {max(0, self.target_gen - self.generation)} 代）。")
                else:
                    self._log("本次目标：持续训练（不设上限）。")
                self.population = []
                for i in range(POPULATION):
                    if self._stop.is_set():
                        raise _Aborted()
                    self.population.append(load_compact(TRAIN_DIR / f"net_{i:03d}"))
                    time.sleep(0.005)  # 主动让出 CPU/GIL，保证控制台按钮可点
                    if (i + 1) % 10 == 0 or i == 0:
                        self._log(f"  加载网络 {i + 1}/{POPULATION}")
                self.current_meta = meta
                self._log(f"从存档继续：已加载第 {self.generation} 代，"
                          f"{len(self.population)} 个网络。")
                return
            except _Aborted:
                raise
            except Exception as e:
                self._log(f"[警告] 读取存档失败（{e}），重新初始化 100 个网络。")
        self._init_population()

    def _init_population(self):
        self.generation = 0
        self.population = []
        self._log("=" * 64)
        self._log(f"未找到 TRAIN 存档，用 data_into 初始化 {POPULATION} 个网络"
                  f"（74884 输入 / 4 输出 / 500 记忆 / 2000 正常）…")
        for i in range(POPULATION):
            if self._stop.is_set():
                return
            cfg, syn, _state = self._data_into_fn(
                input=drn_core.INPUT_COUNT, output=drn_core.OUTPUT_COUNT,
                memory=500, Normal=2000)
            self.population.append(to_compact(cfg, syn))
            del cfg, syn, _state   # 字典只存在一瞬间，避免 100 份字典常驻
            time.sleep(0.005)  # 主动让出 CPU/GIL
            if (i + 1) % 10 == 0:
                self._log(f"  初始化 {i + 1}/{POPULATION}")
        self._save_population()
        self._log(f"初始化完成，已保存到 {TRAIN_DIR}")

    # ---- 主循环 ----

    def _run(self):
        try:
            self._load_or_init()
            write_lock(self.generation, self.target_gen, "running")
            while not self._stop.is_set() and not self._done:
                self._run_generation()
        except _Aborted:
            self._log("[已结束] 已收到结束请求，立即停止（本代不会写入数据库）。")
        except Exception:
            import traceback
            self._log("[错误] 训练线程异常：")
            self._log(traceback.format_exc())
        finally:
            if self._stop.is_set() and not self._done:
                self._log("[已结束] 本次训练已中断，当前未完成的一代不会写入数据库。")
            self._log("训练线程已停止。")
            clear_lock()
            self.sig_finished.emit()

    def _run_generation(self):
        gen = self.generation
        t_gen = time.time()
        self._log("=" * 64)
        self._log(f"===== 第 {gen + 1} 代开始 =====")
        self._log(f"种群 {POPULATION} | 存活 {SURVIVORS} | "
                  f"每存活 {CHILDREN_PER_SURVIVOR} 子代 | 每网络 {TICKS_PER_RUN} tick")

        # 1) 评估全部 100 个网络
        results = []
        self.live_gen = gen
        self.live_results = {}
        for i in range(POPULATION):
            if self._stop.is_set():
                self._log(f"[已结束] 第 {gen + 1} 代评估被中断，本代不写入数据库。")
                return
            traj = Simulator.evaluate(
                None, None, self.screen, ticks=TICKS_PER_RUN,
                cursor_sprite=self.cursor_sprite, compact=self.population[i])
            s = self._score_fn(traj["mouse_positions"], traj["click_positions"])
            clicks = traj["click_positions"]
            hit_w = any(
                43.5 <= x < 77.0 and 676.0 <= y < 712.0 for x, y in clicks
            )
            # 保存回放（供控制台“观看”），按代归档避免进化后对错号
            replay_dir = TRAIN_DIR / "replays" / f"gen_{gen}"
            replay_dir.mkdir(parents=True, exist_ok=True)
            with open(replay_dir / f"net_{i:03d}.json", "w", encoding="utf-8") as f:
                json.dump(traj, f)
            results.append({
                "idx": i, "score": float(s),
                "clicks": len(clicks), "hit_w": bool(hit_w),
                "last_pos": traj["mouse_positions"][-1],
            })
            self.live_results[i] = {
                "score": float(s), "clicks": len(clicks), "hit_w": bool(hit_w),
            }
            self.sig_progress.emit(i + 1, POPULATION, float(s))
            time.sleep(0.005)  # 每个网络之间让出 CPU/GIL，避免界面卡死
            if (i + 1) % 10 == 0 or i == 0:
                self._log(f"  评估 net_{i:03d}  score={s:+.4f}  "
                          f"clicks={len(clicks)}  hit_W={'YES' if hit_w else 'no'}")

        # 2) 打分 → 排序 → 取前 10
        scores = {f"net_{r['idx']:03d}": r["score"] for r in results}
        ranked = self._rank_fn(scores, Top_N=POPULATION)         # 全部排序
        survivors = ranked[:SURVIVORS]                             # 前 10
        survivor_indices = [int(name.split("_")[1]) for name in survivors]

        # 2.5) 多样性检测：只看前 10 名得分；越趋同，变异率越高
        mutation_boost, effective_rates, score_std = adaptive_rates(
            [scores[name] for name in survivors])
        if mutation_boost > 1.0:
            ratio = 1.0 - score_std / CONVERGE_STD
            self._log(f"[多样性] 前10名得分趋同（σ={score_std:.4f}，趋同率 {ratio * 100:.0f}%）"
                      f"→ 变异率 ×{mutation_boost:g}："
                      f"{effective_rates[0]:.4g} ~ {effective_rates[-1]:.4g}")
        else:
            self._log(f"[多样性] 前10名得分分布正常（标准差 {score_std:.4f}），变异率不变。")

        # 3) 构建下一代：10 存活 + 每存活 9 子代
        # 先释放未存活的 90 个网络，给变异阶段的临时字典腾出内存
        keep = set(survivor_indices)
        for i in range(len(self.population)):
            if i not in keep:
                self.population[i] = None
        gc.collect()

        next_pop = []
        meta_list = []
        for si in survivor_indices:
            if self._stop.is_set():
                raise _Aborted()
            next_pop.append(copy_compact(self.population[si]))
            time.sleep(0.005)  # 存活者复制也让出 CPU/GIL
            meta_list.append({
                "name": f"net_{len(next_pop)-1:03d}", "role": "survivor",
                "parent": si, "score": scores[f"net_{si:03d}"],
                "mutation_rate": None,
                "replay_gen": gen, "replay_idx": si,
            })
        for si in survivor_indices:
            # main/mutation.py 需要字典并会 deepcopy；父代字典只在这一轮存在
            cfg_p, syn_p = to_dicts(self.population[si])
            for mr in effective_rates:
                if self._stop.is_set():
                    raise _Aborted()
                cfg_c, syn_c = self._mutation_fn(cfg_p, syn_p, mutation_rate=mr)
                next_pop.append(to_compact(cfg_c, syn_c))
                del cfg_c, syn_c     # 子代立即转回紧凑数组
                time.sleep(0.005)  # 每个子代之间让出 CPU/GIL
                meta_list.append({
                    "name": f"net_{len(next_pop)-1:03d}", "role": "child",
                    "parent": si, "score": None,
                    "mutation_rate": mr,
                    "replay_gen": None, "replay_idx": None,
                })
            del cfg_p, syn_p
            gc.collect()

        # 安全截断到 100
        next_pop = next_pop[:POPULATION]
        meta_list = meta_list[:POPULATION]

        # 4) 详细日志
        score_vals = [r["score"] for r in results]
        best = max(score_vals)
        mean = sum(score_vals) / len(score_vals)
        hit_count = sum(1 for r in results if r["hit_w"])
        self._log("---- 全部 100 个网络得分（从高到低）----")
        for pos, name in enumerate(ranked):
            i = int(name.split("_")[1])
            r = results[i]
            tag = "  ◄ 存活" if name in survivors else ""
            self._log(f"  #{pos+1:>3}  {name}  score={r['score']:+.4f}  "
                      f"clicks={r['clicks']}  hit_W={'YES' if r['hit_w'] else 'no '}"
                      f"  last=({r['last_pos'][0]},{r['last_pos'][1]}){tag}")
        self._log("---- 存活的前 10 个网络 ----")
        for name in survivors:
            self._log(f"  {name}  score={scores[name]:+.4f}  "
                      f"→ 将产生 {CHILDREN_PER_SURVIVOR} 个子代，变异率 {effective_rates}")
        self._log(f"[统计] 最佳={best:+.4f}  平均={mean:+.4f}  "
                  f"命中 W 键={hit_count}/{POPULATION}  耗时={time.time()-t_gen:.1f}s")

        # 5) 结束训练 → 放弃本代（不写数据库）
        if self._stop.is_set():
            self._log(f"[已结束] 第 {gen + 1} 代未完成，已放弃，不写入数据库。")
            return

        # 6) 切换到下一代并保存
        self.population = next_pop
        self.generation = gen + 1
        meta = {
            "generation": self.generation,
            "prev_best": best, "prev_mean": mean, "prev_hit_count": hit_count,
            "duration_s": round(time.time() - t_gen, 1),
            "population": meta_list,
            "ranked": ranked,
            "scores": scores,
            "score_std": round(score_std, 6),
            "mutation_boost": mutation_boost,
            "results": {f"net_{r['idx']:03d}": r for r in results},
        }
        self.current_meta = meta
        self._save_population()
        (TRAIN_DIR / "generation.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log(f"第 {gen+1} 代完成，已演化到第 {self.generation} 代并保存。")
        self.sig_generation_done.emit(meta)
        write_lock(self.generation, self.target_gen, "running")

        if self.target_gen is not None and self.generation >= self.target_gen:
            self._log(f"[完成] 已达到目标代数 {self.target_gen}，本次训练结束。")
            self._done = True

    def _save_population(self):
        for i in range(len(self.population)):
            save_compact(TRAIN_DIR / f"net_{i:03d}", self.population[i])
