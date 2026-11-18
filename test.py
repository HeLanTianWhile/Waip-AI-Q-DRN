# -*- coding: utf-8 -*-
"""
DRN 学习型网络 Demo（持续传播 + 可塑性 + 修剪 + 长程连接）
============================================================
对圆形网络 demo 的改造：
  1. 不再只传播 4 步，而是持续传播，直到网络"变得有规律"（收敛）；
  2. 神经元可塑性（学习规则）：
     - 突触使用频率越高、传递正向信号越多 → W 越大；
     - 传递负（抑制）信号越多 → W 越小；
  3. 突触修剪：|W| 小到极小值 → 裁剪掉；
  4. 神经元修剪：长期不被激活，或持续输出负信号 → 神经元消失（连带剪掉突触）；
  5. 长程连接：信号经常沿"头→…→尾"路径传播 → 头部神经元渐渐长出直达尾部的新突触。

计算规则（Python 与 HTML 内 JS 完全一致）：
    y = Σ(pre.y × W × 递质增益) − B，clamp [-1,1]；运动神经元 act = (y_raw+1)/2 ∈ [0,1]
    ΔW = lr × pre.y × (1 + 使用频率加成)，clamp [-1,1]

运行：python3 drn_learning_demo.py
输出：终端报告 + drn_learning_demo.html（浏览器打开，可交互重跑）
"""

import json
import math
import random

# ---------------- 参数 ----------------

SEED = 42
N_VISUAL, N_AUDITORY, N_MOTOR, N_NORMAL = 5, 5, 10, 80
W_MIN, W_MAX = -0.8, 0.8
B_MIN, B_MAX = -0.3, 0.3
OUT_DEG_MIN, OUT_DEG_MAX = 2, 6

INJECT_PERIOD = 8         # 每 8 步，感官神经元注入一轮 1.0
LR = 0.003                # 可塑性学习率
PRUNE_W = 0.05            # |W| 低于此值 → 裁剪突触
INACTIVE_MAX = 150        # 神经元超过多少步未激活 → 裁剪
NEG_STREAK_MAX = 80       # 神经元连续输出负信号超过多少步 → 裁剪
PATH_CHECK_EVERY = 20     # 每 20 步检查一次长程连接
PATH_COUNT_MIN = 15       # 路径共现计数达到多少 → 新建长程突触
NEW_SY_W = 0.12           # 新长程突触的初始 W
MAX_STEPS = 3000          # 最大步数上限
CONV_WINDOW = 40          # 收敛检测窗口
CONV_EPS = 0.02           # 窗口内各指标相对标准差阈值

# --- P 物质（疼痛）机制 ---
P_INHIBIT = 1.0           # 疼痛抑制强度：电信号最多被打折的比例（0~1，可调）
PAIN_SY_TO_NORMAL = 5     # 疼痛源额外连到的普通神经元数（除全部运动神经元外）
PAIN_DEMO_STEPS = 15      # 收敛后"施加疼痛"对比演示的步数
PAIN_W = 0.8              # 疼痛源突触的初始 W（正 → 学习后疼痛敏化增强）

SIGNAL_TYPES = [
    {"key": "dopamine",  "name": "多巴胺", "gain": 1.15, "color": "#f39c12"},
    {"key": "endorphin", "name": "内啡肽", "gain": 1.00, "color": "#9b59b6"},
    {"key": "serotonin", "name": "血清素", "gain": 0.85, "color": "#1abc9c"},
    {"key": "substanceP", "name": "P物质",  "gain": 1.05, "color": "#f1c40f"},
    {"key": "electric",  "name": "电信号", "gain": 1.00, "color": "#7f8c8d"},
]
NODE_COLORS = {"visual": "#2ecc71", "auditory": "#3498db", "motor": "#e74c3c",
               "normal": "#8892a0", "pain": "#f1c40f"}
NODE_NAMES = {"visual": "视觉", "auditory": "听觉", "motor": "运动", "normal": "普通", "pain": "疼痛源"}


def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


# ---------------- 网络 ----------------

class Neuron:
    __slots__ = ("id", "type", "B", "y", "prev_y", "last_active", "neg_streak", "alive", "external")

    def __init__(self, nid, ntype, B):
        self.id = nid
        self.type = ntype
        self.B = B
        self.y = 0.0
        self.prev_y = 0.0
        self.last_active = 0
        self.neg_streak = 0
        self.alive = True
        self.external = None


class Synapse:
    __slots__ = ("pre", "post", "W", "gain", "signal", "freq", "alive")

    def __init__(self, pre, post, W, gain, signal):
        self.pre = pre
        self.post = post
        self.W = W
        self.gain = gain
        self.signal = signal
        self.freq = 0          # 使用频率（被激活传递的次数）
        self.alive = True


def build_network():
    rng = random.Random(SEED)
    neurons = []
    nid = 0
    for _ in range(N_VISUAL):
        neurons.append(Neuron(nid, "visual", round(rng.uniform(B_MIN, B_MAX), 3))); nid += 1
    for _ in range(N_AUDITORY):
        neurons.append(Neuron(nid, "auditory", round(rng.uniform(B_MIN, B_MAX), 3))); nid += 1
    for _ in range(N_MOTOR):
        neurons.append(Neuron(nid, "motor", round(rng.uniform(B_MIN, B_MAX), 3))); nid += 1
    for _ in range(N_NORMAL):
        neurons.append(Neuron(nid, "normal", round(rng.uniform(B_MIN, B_MAX), 3))); nid += 1

    synapses = []
    for src in neurons:
        out_deg = rng.randint(OUT_DEG_MIN, OUT_DEG_MAX)
        targets = rng.sample([n for n in neurons if n.id != src.id], out_deg)
        for tgt in targets:
            # 运动神经元是执行器：只接收电信号命令（递质不直接驱动运动层）
            if tgt.type == "motor":
                sig = {"key": "electric", "gain": 1.00}
            else:
                sig = rng.choice(SIGNAL_TYPES)
            synapses.append(Synapse(src.id, tgt.id,
                                    round(rng.uniform(W_MIN, W_MAX), 3),
                                    sig["gain"], sig["key"]))

    # 疼痛源神经元（伤害感受器，id=100，放圆心）：
    # 向所有运动神经元 + 若干普通神经元伸出 P 物质突触（疼痛信号通道）
    pain = Neuron(nid, "pain", 0.0)
    neurons.append(pain)
    nid += 1
    for m in neurons:
        if m.type == "motor":
            synapses.append(Synapse(pain.id, m.id, PAIN_W, 1.05, "substanceP"))
    for tgt in rng.sample([n for n in neurons if n.type == "normal"], PAIN_SY_TO_NORMAL):
        synapses.append(Synapse(pain.id, tgt.id, PAIN_W, 1.05, "substanceP"))

    # 运动神经元保证电信号驱动：每个运动神经元从随机普通神经元获得 3 条电信号突触
    # （运动输出由电信号指挥，递质是调制；这样 P 物质抑制电信号才能压住动作）
    motors = [n for n in neurons if n.type == "motor"]
    normals = [n for n in neurons if n.type == "normal"]
    for m in motors:
        for pre in rng.sample(normals, 3):
            synapses.append(Synapse(pre.id, m.id,
                                    round(rng.uniform(0.4, 0.8), 3),
                                    1.00, "electric"))

    return neurons, synapses


def build_in_syn(synapses):
    by_post = {}
    for s in synapses:
        by_post.setdefault(s.post, []).append(s)
    return by_post


# ---------------- 仿真 ----------------

def run_demo(neurons, synapses, max_steps=MAX_STEPS):
    in_syn = build_in_syn(synapses)
    syn_set = {(s.pre, s.post) for s in synapses}
    path_count = {}
    stats = {"del_syn": 0, "del_neu": 0, "new_syn": 0}
    fp_history = []
    converged_at = None
    snapshots = []   # 每 20 步一个快照，供报告与图表

    for t in range(max_steps):
        # 1. 感官周期性注入
        if t % INJECT_PERIOD == 0:
            for n in neurons:
                if n.alive and n.type in ("visual", "auditory"):
                    n.external = 1.0

        # 2. 同步传播（含 P 物质疼痛机制：P 物质抑制本神经元内的电信号）
        for n in neurons:
            n.prev_y = n.y
        new_y = {}
        for n in neurons:
            if not n.alive:
                new_y[n.id] = 0.0
                continue
            if n.type == "pain":
                new_y[n.id] = 0.0      # 疼痛源默认不激活（由疼痛阶段/HTML 开关控制）
                continue
            if n.external is not None:
                new_y[n.id] = 1.0
            else:
                elec = oth = p = 0.0
                for s in in_syn.get(n.id, ()):
                    if s.alive:
                        pre = neurons[s.pre]
                        if pre.alive:
                            if s.signal == "electric":
                                elec += pre.prev_y * s.W
                            elif s.signal == "substanceP":
                                p += pre.prev_y                 # 疼痛强度（P 物质传入量）
                            else:
                                oth += pre.prev_y * s.W * s.gain
                k = min(1.0, max(0.0, p)) * P_INHIBIT           # 抑制系数
                raw = clamp(oth + elec * (1.0 - k) - n.B)        # 电信号被打折 (1-k)
                new_y[n.id] = (raw + 1.0) / 2.0 if n.type == "motor" else raw
        for n in neurons:
            if n.alive:
                n.y = new_y[n.id]
                n.external = None

        # 3. 可塑性：频率越高/正向越多 → W 越大；负信号多 → W 越小
        for s in synapses:
            if not s.alive:
                continue
            pre = neurons[s.pre]
            if not pre.alive:
                continue
            if abs(pre.y) > 0.01:
                boost = min(1.0, s.freq / 50.0)          # 使用频率加成
                s.W = clamp(s.W + LR * pre.y * (1.0 + boost))
                s.freq += 1

        # 4. 长程连接累积：两步链 c→a→b 共现
        for s in synapses:
            if not s.alive:
                continue
            a = neurons[s.pre]
            b = neurons[s.post]
            if not a.alive or not b.alive:
                continue
            if a.y > 0.05 and b.y > 0.05 and a.prev_y > 0.05:
                for s2 in in_syn.get(a.id, ()):
                    if not s2.alive:
                        continue
                    c = neurons[s2.pre]
                    if c.alive and c.prev_y > 0.05 and c.id != b.id:
                        key = (c.id, b.id)
                        path_count[key] = path_count.get(key, 0) + 1

        # 5. 周期检查：路径共现足够 → 新建长程突触
        if t % PATH_CHECK_EVERY == PATH_CHECK_EVERY - 1:
            for (c_id, b_id), cnt in list(path_count.items()):
                if cnt >= PATH_COUNT_MIN and (c_id, b_id) not in syn_set:
                    b_neuron = neurons[b_id]
                    if b_neuron.type == "motor":
                        sig = {"key": "electric", "gain": 1.00}
                    else:
                        sig = random.choice(SIGNAL_TYPES)
                    synapses.append(Synapse(c_id, b_id, NEW_SY_W, sig["gain"], sig["key"]))
                    in_syn.setdefault(b_id, []).append(synapses[-1])
                    syn_set.add((c_id, b_id))
                    stats["new_syn"] += 1
                path_count[(c_id, b_id)] = 0

        # 6. 修剪突触：|W| 极小 → 裁剪
        for s in synapses:
            if s.alive and abs(s.W) < PRUNE_W:
                s.alive = False
                stats["del_syn"] += 1

        # 7. 修剪神经元：长期不激活 / 持续负输出 → 消失（连带剪突触）
        #    感官输入与疼痛源永不裁剪
        for n in neurons:
            if not n.alive or n.type in ("visual", "auditory", "pain"):
                continue
            if n.y < -0.05:
                n.neg_streak += 1
            if abs(n.y) > 0.05:
                n.last_active = t
                n.neg_streak = 0
            if t - n.last_active > INACTIVE_MAX or n.neg_streak > NEG_STREAK_MAX:
                n.alive = False
                stats["del_neu"] += 1
                for s in synapses:
                    if s.alive and (s.pre == n.id or s.post == n.id):
                        s.alive = False
                        stats["del_syn"] += 1

        # 8. 收敛检测：近期窗口内指标相对稳定 → 传播"变得有规律"
        alive = [n for n in neurons if n.alive]
        syns = [s for s in synapses if s.alive]
        avg_abs_y = sum(abs(n.y) for n in alive) / len(alive) if alive else 0.0
        avg_w = sum(abs(s.W) for s in syns) / len(syns) if syns else 0.0
        fp_history.append((avg_abs_y, avg_w, len(alive), len(syns)))

        if t % 20 == 19 and len(fp_history) >= CONV_WINDOW:
            recent = fp_history[-CONV_WINDOW:]
            cols = list(zip(*recent))
            stable = True
            for col in cols:
                mean = sum(col) / len(col)
                if mean == 0:
                    continue
                var = sum((x - mean) ** 2 for x in col) / len(col)
                if (var ** 0.5) / mean > CONV_EPS:
                    stable = False
                    break
            if stable:
                converged_at = t
                break

        if t % 20 == 19:
            snapshots.append({"t": t, "neu": len(alive), "syn": len(syns),
                              "avgW": round(avg_w, 3), "avgAbsY": round(avg_abs_y, 3),
                              **{k: stats[k] for k in stats}})

    if converged_at is None:
        converged_at = max_steps - 1
    snapshots.append({"t": converged_at, "neu": len([n for n in neurons if n.alive]),
                      "syn": len([s for s in synapses if s.alive]),
                      "avgW": round(avg_w, 3), "avgAbsY": round(avg_abs_y, 3),
                      **{k: stats[k] for k in stats}})

    # --- 疼痛对比演示：收敛状态下，无疼痛 vs 施加疼痛（只传播，不学习不修剪）---
    def avg_motor():
        ms = [n for n in neurons if n.alive and n.type == "motor"]
        return sum(n.y for n in ms) / len(ms) if ms else 0.0

    pain_before_list = {n.id: round(n.y, 3) for n in neurons
                        if n.alive and n.type == "motor"}
    pain_before = round(avg_motor(), 3)
    for _ in range(PAIN_DEMO_STEPS):
        for n in neurons:
            n.prev_y = n.y
        new_y = {}
        for n in neurons:
            if not n.alive:
                new_y[n.id] = 0.0
                continue
            if n.type == "pain":
                new_y[n.id] = 1.0          # 疼痛源激活 → 向运动神经元释放 P 物质
                continue
            if n.external is not None:
                new_y[n.id] = 1.0
            else:
                elec = oth = p = 0.0
                for s in in_syn.get(n.id, ()):
                    if s.alive:
                        pre = neurons[s.pre]
                        if pre.alive:
                            if s.signal == "electric":
                                elec += pre.prev_y * s.W
                            elif s.signal == "substanceP":
                                p += pre.prev_y
                            else:
                                oth += pre.prev_y * s.W * s.gain
                k = min(1.0, max(0.0, p)) * P_INHIBIT
                raw = clamp(oth + elec * (1.0 - k) - n.B)
                new_y[n.id] = (raw + 1.0) / 2.0 if n.type == "motor" else raw
        for n in neurons:
            if n.alive:
                n.y = new_y[n.id]
                n.external = None
    pain_after = round(avg_motor(), 3)
    pain_after_list = {n.id: round(n.y, 3) for n in neurons
                       if n.alive and n.type == "motor"}
    per_neuron = [(nid, pain_before_list.get(nid, 0.0), pain_after_list.get(nid, 0.0))
                  for nid in sorted(set(pain_before_list) | set(pain_after_list))]
    pain_compare = {"before": pain_before, "after": pain_after,
                    "steps": PAIN_DEMO_STEPS, "per_neuron": per_neuron}

    return converged_at, stats, snapshots, pain_compare


# ---------------- 终端报告 ----------------

def report(neurons, synapses, converged_at, stats, snapshots, init_syn_count, pain_compare):
    print("=" * 64)
    print("DRN 学习型网络 Demo —— 持续传播 + 可塑性 + 修剪 + 长程连接")
    print("=" * 64)

    alive = [n for n in neurons if n.alive]
    syns = [s for s in synapses if s.alive]
    print(f"\n[初始] 100 神经元（5视觉/5听觉/10运动/80普通），"
          f"{init_syn_count} 条随机突触")

    print(f"\n[运行] 共传播 {converged_at + 1} 步后收敛"
          f"{'（达到步数上限）' if converged_at == MAX_STEPS - 1 else '：传播已变得有规律'}")

    print("\n[过程快照] 每 200 步")
    print("   步数 | 存活神经元 | 突触数 | |W|均值 | 已剪突触 | 已剪神经元 | 新长程连接")
    for sp in snapshots:
        if sp["t"] % 200 == 0 or sp["t"] == snapshots[-1]["t"]:
            print(f"  {sp['t']:>5} | {sp['neu']:>6} | {sp['syn']:>5} | "
                  f"{sp['avgW']:>6.3f} | {sp['del_syn']:>6} | {sp['del_neu']:>6} | {sp['new_syn']:>8}")

    print(f"\n[最终网络] 存活神经元 {len(alive)}/100，突触 {len(syns)} 条")
    tcount = {}
    for n in alive:
        tcount[n.type] = tcount.get(n.type, 0) + 1
    for t in ("visual", "auditory", "motor", "normal"):
        print(f"  {NODE_NAMES[t]:<4}：{tcount.get(t, 0)} 个")

    w_vals = [s.W for s in syns]
    print(f"\n[可塑性结果] W 范围 [{min(w_vals):.2f}, {max(w_vals):.2f}]，"
          f"其中 |W|≥0.5 的突触 {sum(1 for w in w_vals if abs(w) >= 0.5)} 条（被强化）")
    print(f"  被裁剪突触 {stats['del_syn']} 条，被裁剪神经元 {stats['del_neu']} 个，"
          f"新建长程连接 {stats['new_syn']} 条")

    motors = sorted([n for n in alive if n.type == "motor"], key=lambda n: -n.y)
    if motors:
        print("\n[最终运动神经元输出] 无疼痛 → 施加疼痛（P 物质压住电信号）")
        for nid, before, after in pain_compare["per_neuron"]:
            arrow = "↓" if after < before else ("→" if after == before else "↑")
            print(f"  运动#{nid:<3} act {before:.3f} {arrow} {after:.3f}")

    # P 物质（疼痛）机制演示结果
    print(f"\n[P 物质疼痛机制] 疼痛源（伤害感受器）通过 P 物质突触连接到所有运动神经元")
    print(f"  无疼痛时运动输出平均：{pain_compare['before']:.3f}")
    print(f"  施加疼痛 {pain_compare['steps']} 步后运动输出平均：{pain_compare['after']:.3f}")
    drop = pain_compare['before'] - pain_compare['after']
    print(f"  运动输出下降：{drop:.3f}"
          f"（P 物质进入运动神经元 → 抑制其内部电信号 → 动作被压制）")
    print(f"  注：P 物质突触随使用会变粗（疼痛敏化），越疼越抑制；"
          f"抑制强度系数 P_INHIBIT={P_INHIBIT} 可调")
    print("\n  结论：网络在持续刺激下一边传播一边学习，"
          "弱突触被剪掉、强通路被保留、路径头部长出直达尾部的长程连接，最终传播趋于规律。")


# ---------------- HTML 模板 ----------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230d1117'/%3E%3Cpath d='M32 32 L12 16 M32 32 L52 14 M32 32 L52 50 M32 32 L12 50' stroke='%2358a6ff' stroke-width='2' opacity='0.7'/%3E%3Ccircle cx='32' cy='32' r='10' fill='%2358a6ff'/%3E%3Ccircle cx='12' cy='16' r='5' fill='%232ecc71'/%3E%3Ccircle cx='52' cy='14' r='5' fill='%233498db'/%3E%3Ccircle cx='52' cy='50' r='5' fill='%23e74c3c'/%3E%3Ccircle cx='12' cy='50' r='5' fill='%238892a0'/%3E%3C/svg%3E">
<title>DRN 学习型网络 Demo</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2333;--line:#2d333b;
        --text:#e6edf3;--dim:#8b949e;--acc:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:"Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif}
  .wrap{max-width:1360px;margin:0 auto;padding:16px}
  header{padding:8px 4px 12px}
  h1{font-size:22px;font-weight:700}
  .sub{color:var(--dim);font-size:14px;margin-top:6px;line-height:1.7}
  .sub b{color:var(--acc);font-weight:600}
  .layout{display:grid;grid-template-columns:1fr 330px;gap:14px}
  @media(max-width:960px){.layout{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
  svg#net{width:100%;height:auto;display:block;background:radial-gradient(circle at 50% 42%,#131a26 0%,#0b0f16 72%)}
  .legend{display:flex;flex-wrap:wrap;gap:8px 14px;padding:8px 2px 2px;font-size:12px;color:var(--dim)}
  .legend .item{display:flex;align-items:center;gap:6px}
  .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
  .swatch{width:16px;height:4px;border-radius:2px;display:inline-block}
  .controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
  button{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:8px;
         padding:9px 14px;font-size:13px;cursor:pointer;transition:.15s;min-height:40px}
  button:hover{border-color:var(--acc);color:#fff}
  button.active{background:var(--acc);color:#0d1117;border-color:var(--acc);font-weight:600}
  .speed{margin:2px 0 10px;font-size:12px;color:var(--dim);display:flex;align-items:center;gap:8px}
  .speed .val{color:var(--acc);font-weight:700;font-variant-numeric:tabular-nums}
  .state-line{display:flex;align-items:center;gap:8px;font-size:13px;margin:2px 0 8px}
  .pill{padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600}
  .pill.run{background:rgba(88,166,255,.15);color:var(--acc);border:1px solid var(--acc)}
  .pill.done{background:rgba(63,185,80,.15);color:var(--good);border:1px solid var(--good)}
  .stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .stat{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
  .stat .k{font-size:11px;color:var(--dim)}
  .stat .v{font-size:16px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
  .stat .v.small{font-size:13px}
  h3{font-size:13px;color:var(--dim);font-weight:600;margin:14px 0 8px;letter-spacing:.5px}
  #trend{width:100%;height:110px;display:block;background:var(--panel2);border:1px solid var(--line);border-radius:8px}
  .motor-list{margin-top:4px}
  .motor-row{display:flex;align-items:center;gap:8px;margin:4px 0}
  .motor-row .nm{width:58px;font-size:11px;color:var(--dim);flex:none}
  .motor-bar{flex:1;height:9px;background:var(--panel2);border-radius:5px;overflow:hidden;border:1px solid var(--line)}
  .motor-fill{height:100%;width:0%;background:linear-gradient(90deg,#e74c3c,#ff6b6b);border-radius:5px;transition:width .25s}
  .motor-val{width:44px;text-align:right;font-size:11px;font-variant-numeric:tabular-nums}
  .rules{font-size:12px;color:var(--dim);line-height:1.9;margin-top:12px;border-top:1px dashed var(--line);padding-top:10px}
  .rules b{color:var(--text)}
  footer{color:#484f58;font-size:11px;text-align:center;padding:16px 0 6px}
  .dim{color:var(--dim)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>DRN 学习型网络 Demo</h1>
    <div class="sub">持续传播直到<b>变得有规律</b>。感官每 8 步注入一轮信号；突触<b>频率越高、正向越多 → W 越大</b>，负信号多 → W 越小；
      |W| 极小 → 剪突触；神经元长期不激活 / 持续负输出 → 消失；
      路径"头→…→尾"经常共现 → 头部渐渐长出直达尾部的新突触。</div>
  </header>

  <div class="layout">
    <div class="card">
      <svg id="net" viewBox="0 0 800 800" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <radialGradient id="halo" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#ffffff" stop-opacity="0.9"/>
            <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
          </radialGradient>
        </defs>
        <g id="edges"></g>
        <g id="nodes"></g>
      </svg>
      <div class="legend" id="legend"></div>
    </div>

    <div>
      <div class="card">
        <div class="controls">
          <button id="btnPlay">播放</button>
          <button id="btnStep">单步</button>
          <button id="btnJump">直达收敛</button>
          <button id="btnReset">重置</button>
          <button id="btnPain" title="疼痛源激活 → P 物质进入神经元 → 特异抑制神经元内的电信号（其他递质不受影响）">施加疼痛</button>
        </div>
        <div class="speed">播放速度：<span class="val" id="spdVal">1×</span>
          <button id="btnSpeed" style="min-height:30px;padding:4px 10px">加速</button></div>
        <div class="state-line">状态：<span class="pill run" id="statePill">运行中</span><span class="dim" id="stateTxt">t=0</span></div>
        <div class="stat-grid">
          <div class="stat"><div class="k">步数</div><div class="v" id="stT">0</div></div>
          <div class="stat"><div class="k">存活神经元</div><div class="v" id="stNeu">-</div></div>
          <div class="stat"><div class="k">突触数</div><div class="v" id="stSyn">-</div></div>
          <div class="stat"><div class="k">|W| 均值</div><div class="v small" id="stW">-</div></div>
          <div class="stat"><div class="k">已剪突触</div><div class="v small" id="stDelS">0</div></div>
          <div class="stat"><div class="k">已剪神经元</div><div class="v small" id="stDelN">0</div></div>
          <div class="stat"><div class="k">新长程连接</div><div class="v small" id="stNew">0</div></div>
          <div class="stat"><div class="k">激活神经元</div><div class="v small" id="stAct">-</div></div>
        </div>
        <h3>结构演化（突触数 / 神经元数）</h3>
        <svg id="trend" viewBox="0 0 300 110" preserveAspectRatio="none">
          <line x1="0" y1="95" x2="300" y2="95" stroke="#2d333b" stroke-width="1"/>
          <line x1="0" y1="5" x2="300" y2="5" stroke="#2d333b" stroke-width="1"/>
          <polyline id="trendSyn" fill="none" stroke="#58a6ff" stroke-width="1.6" points=""/>
          <polyline id="trendNeu" fill="none" stroke="#3fb950" stroke-width="1.6" points=""/>
        </svg>
        <h3>运动神经元输出（act ∈ [0,1]）</h3>
        <div class="motor-list" id="motorList"></div>
        <div class="rules">
          <b>学习规则：</b><br>
          1. 突触使用频率越高、传递正向信号越多 → W 越大；负信号越多 → W 越小<br>
          2. |W| 小于 0.05 → 突触被裁剪<br>
          3. 神经元 >150 步未激活，或连续 >80 步输出负信号 → 神经元消失（感官输入不裁剪）<br>
          4. 路径"头→…→尾"反复共现 → 头部长出直达尾部的新突触（捷径）<br>
          5. <b>P 物质（疼痛）</b>：圆心疼痛源激活 → P 物质进入神经元 → 只压住该神经元内的<b>电信号</b>（多巴胺/内啡肽/血清素不受影响）→ 运动输出被压制，越疼越动不了<br>
          6. 网络持续运行，直到传播变得有规律（近期状态指标稳定）即收敛
        </div>
      </div>
    </div>
  </div>
  <footer>DRN 量化版 · 学习型网络原型演示（随机初始连接，示意性质）· 疼痛源 = 圆心黄点</footer>
</div>

<script>
"use strict";
const DATA = __DATA__;
const N = DATA.neurons.length;
const INJECT = DATA.injectPeriod, LR = DATA.lr, PRUNE_W = DATA.pruneW,
      INACTIVE_MAX = DATA.inactiveMax, NEG_MAX = DATA.negMax,
      PATH_EVERY = DATA.pathEvery, PATH_MIN = DATA.pathMin,
      MAX_STEPS = DATA.maxSteps, CONV_WIN = DATA.convWindow, CONV_EPS = DATA.convEps;

// ---- 状态 ----
const neurons = DATA.neurons.map(n => ({...n, y:0, prev_y:0, last_active:0, neg_streak:0, alive:true, external:null}));
const synapses = DATA.synapses.map(s => ({...s, freq:0, alive:true}));
const inSyn = {};
synapses.forEach(s => { (inSyn[s.post] = inSyn[s.post] || []).push(s); });
const synSet = new Set(synapses.map(s => s.pre + "_" + s.post));
const pathCount = {};
const stats = {delSyn:0, delNeu:0, newSyn:0};
let PAIN_ON = false;
const P_INHIBIT = DATA.pInhibit;
let t = 0, converged = false, running = false, timer = null, speed = 1;
const SPEEDS = [1, 4, 16, 64];
let trend = [];
const fpHist = [];

function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

// ---- 布局（疼痛源放圆心，其余 100 个在圆周）----
const CX = 400, CY = 400, R = 340;
const POS = neurons.map((n, i) => {
  if (n.type === "pain") return {x: CX, y: CY};
  const a = (i / (N - 1)) * Math.PI * 2 - Math.PI / 2;
  return {x: CX + R * Math.cos(a), y: CY + R * Math.sin(a)};
});

// ---- 建 SVG ----
const NS = "http://www.w3.org/2000/svg";
const gEdges = document.getElementById("edges");
const gNodes = document.getElementById("nodes");
const edgeEls = synapses.map(s => {
  const p1 = POS[s.pre], p2 = POS[s.post];
  const mx = (p1.x + p2.x)/2, my = (p1.y + p2.y)/2;
  const dx = mx - CX, dy = my - CY;
  const dl = Math.sqrt(dx*dx + dy*dy) || 1;
  const k = 22 + Math.abs(s.W) * 30;
  const cxp = mx + dx/dl*k, cyp = my + dy/dl*k;
  const path = document.createElementNS(NS, "path");
  path.setAttribute("d", `M${p1.x.toFixed(1)} ${p1.y.toFixed(1)} Q${cxp.toFixed(1)} ${cyp.toFixed(1)} ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`);
  const info = DATA.signalTypes.find(x => x.key === s.signal);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", info.color);
  path.setAttribute("stroke-width", (0.4 + Math.abs(s.W) * 1.5).toFixed(2));
  path.setAttribute("opacity", 0.14);
  gEdges.appendChild(path);
  return path;
});
const nodeEls = neurons.map(n => {
  const p = POS[n.id];
  const g = document.createElementNS(NS, "g");
  const halo = document.createElementNS(NS, "circle");
  halo.setAttribute("cx", p.x); halo.setAttribute("cy", p.y);
  halo.setAttribute("r", 6); halo.setAttribute("fill", "url(#halo)"); halo.setAttribute("opacity", 0);
  const c = document.createElementNS(NS, "circle");
  c.setAttribute("cx", p.x); c.setAttribute("cy", p.y);
  c.setAttribute("r", n.type === "normal" ? 5 : (n.type === "pain" ? 10 : 7));
  c.setAttribute("fill", DATA.nodeColors[n.type]);
  c.setAttribute("stroke", "#0d1117"); c.setAttribute("stroke-width", 1.5);
  g.appendChild(halo); g.appendChild(c);
  gNodes.appendChild(g);
  return {g, halo, c, n};
});

// ---- 图例 ----
const legendEl = document.getElementById("legend");
Object.entries(DATA.nodeNames).forEach(([key, name]) => {
  const item = document.createElement("span");
  item.className = "item";
  item.innerHTML = `<span class="dot" style="background:${DATA.nodeColors[key]}"></span>${name}`;
  legendEl.appendChild(item);
});
DATA.signalTypes.forEach(si => {
  const item = document.createElement("span");
  item.className = "item";
  item.innerHTML = `<span class="swatch" style="background:${si.color}"></span>${si.name}`;
  legendEl.appendChild(item);
});

// ---- 仿真核心（与 Python 一致）----
function stepOnce(){
  if (converged && !PAIN_ON) return;
  // 1. 感官注入
  if (t % INJECT === 0){
    neurons.forEach(n => { if (n.alive && (n.type === "visual" || n.type === "auditory")) n.external = 1.0; });
  }
  // 2. 同步传播（含 P 物质疼痛机制：P 物质特异抑制本神经元内的电信号）
  neurons.forEach(n => n.prev_y = n.y);
  const newY = new Array(N).fill(0);
  neurons.forEach(n => {
    if (!n.alive){ newY[n.id] = 0; return; }
    if (n.type === "pain"){ newY[n.id] = PAIN_ON ? 1.0 : 0.0; return; }
    if (n.external !== null){ newY[n.id] = 1.0; return; }
    let elec = 0, oth = 0, p = 0;
    (inSyn[n.id] || []).forEach(s => {
      if (!s.alive || !neurons[s.pre].alive) return;
      const preY = neurons[s.pre].prev_y;
      if (s.signal === "electric") elec += preY * s.W;
      else if (s.signal === "substanceP") p += preY;
      else oth += preY * s.W * s.gain;
    });
    const k = Math.min(1, Math.max(0, p)) * P_INHIBIT;
    const raw = clamp(oth + elec * (1 - k) - n.B, -1, 1);
    newY[n.id] = n.type === "motor" ? (raw + 1) / 2 : raw;
  });
  neurons.forEach(n => { if (n.alive){ n.y = newY[n.id]; n.external = null; } });

  // 3. 可塑性
  synapses.forEach(s => {
    if (!s.alive) return;
    const pre = neurons[s.pre];
    if (!pre.alive) return;
    if (Math.abs(pre.y) > 0.01){
      const boost = Math.min(1, s.freq / 50);
      s.W = clamp(s.W + LR * pre.y * (1 + boost), -1, 1);
      s.freq++;
    }
  });

  // 4. 长程连接累积
  synapses.forEach(s => {
    if (!s.alive) return;
    const a = neurons[s.pre], b = neurons[s.post];
    if (!a.alive || !b.alive) return;
    if (a.y > 0.05 && b.y > 0.05 && a.prev_y > 0.05){
      (inSyn[a.id] || []).forEach(s2 => {
        if (!s2.alive) return;
        const c = neurons[s2.pre];
        if (c.alive && c.prev_y > 0.05 && c.id !== b.id){
          const key = c.id + "_" + b.id;
          pathCount[key] = (pathCount[key] || 0) + 1;
        }
      });
    }
  });

  // 5. 周期检查：新建长程突触
  if (t % PATH_EVERY === PATH_EVERY - 1){
    Object.keys(pathCount).forEach(key => {
      if (pathCount[key] >= PATH_MIN && !synSet.has(key)){
        const [cid, bid] = key.split("_").map(Number);
        // 运动神经元只接收电信号命令（与 Python 一致）
        const isMotor = neurons[bid].type === "motor";
        const info = isMotor
          ? DATA.signalTypes.find(x => x.key === "electric")
          : DATA.signalTypes[Math.floor(Math.random() * DATA.signalTypes.length)];
        const ns = {pre: cid, post: bid, W: 0.12, gain: info.gain, signal: info.key, freq: 0, alive: true};
        synapses.push(ns);
        (inSyn[bid] = inSyn[bid] || []).push(ns);
        synSet.add(key);
        stats.newSyn++;
        // 追加边元素
        const p1 = POS[cid], p2 = POS[bid];
        const mx = (p1.x + p2.x)/2, my = (p1.y + p2.y)/2;
        const dx = mx - CX, dy = my - CY;
        const dl = Math.sqrt(dx*dx + dy*dy) || 1;
        const cxp = mx + dx/dl*24, cyp = my + dy/dl*24;
        const path = document.createElementNS(NS, "path");
        path.setAttribute("d", `M${p1.x.toFixed(1)} ${p1.y.toFixed(1)} Q${cxp.toFixed(1)} ${cyp.toFixed(1)} ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", info.color);
        path.setAttribute("stroke-width", "0.8");
        path.setAttribute("opacity", "0.5");
        gEdges.appendChild(path);
        edgeEls.push(path);
      }
      pathCount[key] = 0;
    });
  }

  // 6. 修剪突触
  synapses.forEach(s => {
    if (s.alive && Math.abs(s.W) < PRUNE_W){ s.alive = false; stats.delSyn++; }
  });

  // 7. 修剪神经元
  neurons.forEach(n => {
    if (!n.alive || n.type === "visual" || n.type === "auditory" || n.type === "pain") return;
    if (n.y < -0.05) n.neg_streak++;
    if (Math.abs(n.y) > 0.05){ n.last_active = t; n.neg_streak = 0; }
    if (t - n.last_active > INACTIVE_MAX || n.neg_streak > NEG_MAX){
      n.alive = false;
      stats.delNeu++;
      synapses.forEach(s => {
        if (s.alive && (s.pre === n.id || s.post === n.id)){ s.alive = false; stats.delSyn++; }
      });
    }
  });

  t++;

  // 8. 收敛检测
  const alive = neurons.filter(n => n.alive);
  const syns = synapses.filter(s => s.alive);
  const avgAbsY = alive.length ? alive.reduce((a, n) => a + Math.abs(n.y), 0) / alive.length : 0;
  const avgW = syns.length ? syns.reduce((a, s) => a + Math.abs(s.W), 0) / syns.length : 0;
  const fp = [avgAbsY, avgW, alive.length, syns.length];
  fpHist.push(fp);
  if (t % 20 === 19 && fpHist.length >= CONV_WIN){
    const recent = fpHist.slice(-CONV_WIN);
    let stable = true;
    for (let c = 0; c < 4; c++){
      const col = recent.map(r => r[c]);
      const mean = col.reduce((a, x) => a + x, 0) / col.length;
      if (mean === 0) continue;
      const varv = col.reduce((a, x) => a + (x - mean) ** 2, 0) / col.length;
      if (Math.sqrt(varv) / mean > CONV_EPS){ stable = false; break; }
    }
    if (stable) converged = true;
  }
  if (t >= MAX_STEPS) converged = true;

  // 趋势点
  if (t % 10 === 0 || converged){
    trend.push({t, syn: synapses.filter(s => s.alive).length, neu: neurons.filter(n => n.alive).length});
    if (trend.length > 400) trend = trend.slice(-400);
  }
}

// ---- 渲染 ----
function render(){
  // 节点
  nodeEls.forEach(({halo, c, n}) => {
    if (!n.alive){ c.setAttribute("opacity", 0.06); halo.setAttribute("opacity", 0); return; }
    c.setAttribute("opacity", 1);
    const y = n.y, act = Math.abs(y);
    let fill = DATA.nodeColors[n.type];
    if (y > 0.05) fill = "#ffffff";
    else if (y < -0.05) fill = "#a78bfa";
    c.setAttribute("fill", fill);
    const baseR = n.type === "normal" ? 5 : (n.type === "pain" ? 10 : 7);
    c.setAttribute("r", (baseR + act * 4).toFixed(1));
    halo.setAttribute("opacity", Math.min(0.85, act * 1.6).toFixed(2));
    halo.setAttribute("r", (6 + act * 14).toFixed(1));
    c.setAttribute("stroke", y > 0.05 ? "#ffffff" : "#0d1117");
  });
  // 边
  edgeEls.forEach((path, i) => {
    const s = synapses[i];
    if (!s || !s.alive){ path.setAttribute("opacity", 0.03); return; }
    const pre = neurons[s.pre];
    const on = pre.alive && Math.abs(pre.y) > 0.05;
    const info = DATA.signalTypes.find(x => x.key === s.signal);
    path.setAttribute("opacity", on ? Math.min(0.9, 0.3 + Math.abs(s.W)) : 0.14);
    path.setAttribute("stroke-width", (0.4 + Math.abs(s.W) * 1.5).toFixed(2));
    path.setAttribute("stroke-dasharray", on ? "6 5" : "none");
  });
  // 面板
  const alive = neurons.filter(n => n.alive);
  const syns = synapses.filter(s => s.alive);
  const actCnt = alive.filter(n => Math.abs(n.y) > 0.05).length;
  const avgW = syns.length ? syns.reduce((a, s) => a + Math.abs(s.W), 0) / syns.length : 0;
  document.getElementById("stT").textContent = t;
  document.getElementById("stNeu").textContent = alive.length + " / " + N;
  document.getElementById("stSyn").textContent = syns.length;
  document.getElementById("stW").textContent = avgW.toFixed(3);
  document.getElementById("stDelS").textContent = stats.delSyn;
  document.getElementById("stDelN").textContent = stats.delNeu;
  document.getElementById("stNew").textContent = stats.newSyn;
  document.getElementById("stAct").textContent = actCnt;
  const pill = document.getElementById("statePill");
  if (converged){
    pill.className = "pill done"; pill.textContent = "已收敛";
    document.getElementById("stateTxt").textContent = "传播已变得有规律，共 " + t + " 步（可点「施加疼痛」看动作被压制）";
    if (timer){ clearInterval(timer); timer = null; running = false; document.getElementById("btnPlay").textContent = "播放"; }
  } else {
    pill.className = "pill run"; pill.textContent = "运行中";
    document.getElementById("stateTxt").textContent = "正在学习与修剪";
  }
  // 运动输出
  const ml = document.getElementById("motorList");
  ml.innerHTML = "";
  neurons.filter(n => n.type === "motor" && n.alive)
    .sort((a, b) => b.y - a.y)
    .forEach(n => {
      const row = document.createElement("div");
      row.className = "motor-row";
      row.innerHTML = `<span class="nm">运动#${n.id}</span>
        <div class="motor-bar"><div class="motor-fill" style="width:${(n.y * 100).toFixed(0)}%"></div></div>
        <span class="motor-val">${n.y.toFixed(3)}</span>`;
      ml.appendChild(row);
    });
  // 趋势图
  if (trend.length >= 2){
    const W = 300, H = 110;
    const maxT = Math.max(t, 1);
    const maxVal = Math.max(420, ...trend.map(p => p.syn));
    const ptsSyn = trend.map(p => `${(p.t / maxT * W).toFixed(1)},${(95 - p.syn / maxVal * 85).toFixed(1)}`).join(" ");
    const ptsNeu = trend.map(p => `${(p.t / maxT * W).toFixed(1)},${(95 - p.neu / maxVal * 85).toFixed(1)}`).join(" ");
    document.getElementById("trendSyn").setAttribute("points", ptsSyn);
    document.getElementById("trendNeu").setAttribute("points", ptsNeu);
  }
}

// ---- 控制 ----
const btnPlay = document.getElementById("btnPlay");
const btnStep = document.getElementById("btnStep");
const btnJump = document.getElementById("btnJump");
const btnReset = document.getElementById("btnReset");
const btnSpeed = document.getElementById("btnSpeed");
const btnPain = document.getElementById("btnPain");
const spdVal = document.getElementById("spdVal");

function togglePlay(){
  if (running){ clearInterval(timer); timer = null; running = false; btnPlay.textContent = "播放"; }
  else {
    running = true; btnPlay.textContent = "暂停";
    const tick = () => {
      if (converged && !PAIN_ON){ togglePlay(); render(); return; }
      for (let i = 0; i < speed; i++) stepOnce();
      render();
      timer = setTimeout(tick, 90);
    };
    timer = setTimeout(tick, 90);
  }
}
btnPlay.addEventListener("click", togglePlay);
btnStep.addEventListener("click", () => { if (!converged || PAIN_ON){ for (let i = 0; i < speed; i++) stepOnce(); render(); } });
btnJump.addEventListener("click", () => {
  if (timer){ clearInterval(timer); timer = null; running = false; btnPlay.textContent = "播放"; }
  while (!converged) stepOnce();
  render();
});
btnReset.addEventListener("click", () => {
  if (timer){ clearInterval(timer); timer = null; running = false; btnPlay.textContent = "播放"; }
  PAIN_ON = false;
  btnPain.classList.remove("active");
  btnPain.textContent = "施加疼痛";
  neurons.forEach(n => { n.y = 0; n.prev_y = 0; n.last_active = 0; n.neg_streak = 0; n.alive = true; n.external = null; });
  synapses.forEach(s => { s.freq = 0; s.alive = true; });
  synapses.length = DATA.synapses.length;
  DATA.synapses.forEach((s, i) => { synapses[i].pre = s.pre; synapses[i].post = s.post; synapses[i].W = s.W; synapses[i].gain = s.gain; synapses[i].signal = s.signal; });
  while (gEdges.firstChild) gEdges.removeChild(gEdges.firstChild);
  edgeEls.length = 0;
  DATA.synapses.forEach(s => {
    const p1 = POS[s.pre], p2 = POS[s.post];
    const mx = (p1.x + p2.x)/2, my = (p1.y + p2.y)/2;
    const dx = mx - CX, dy = my - CY;
    const dl = Math.sqrt(dx*dx + dy*dy) || 1;
    const cxp = mx + dx/dl*24, cyp = my + dy/dl*24;
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", `M${p1.x.toFixed(1)} ${p1.y.toFixed(1)} Q${cxp.toFixed(1)} ${cyp.toFixed(1)} ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`);
    const info = DATA.signalTypes.find(x => x.key === s.signal);
    path.setAttribute("fill", "none"); path.setAttribute("stroke", info.color);
    path.setAttribute("stroke-width", (0.4 + Math.abs(s.W) * 1.5).toFixed(2));
    path.setAttribute("opacity", 0.14);
    gEdges.appendChild(path);
    edgeEls.push(path);
  });
  t = 0; converged = false; stats.delSyn = 0; stats.delNeu = 0; stats.newSyn = 0;
  trend = []; fpHist.length = 0;
  Object.keys(pathCount).forEach(k => delete pathCount[k]);
  // 重建索引（去掉运行期新建的长程突触）
  Object.keys(inSyn).forEach(k => delete inSyn[k]);
  synapses.forEach(s => { (inSyn[s.post] = inSyn[s.post] || []).push(s); });
  synSet.clear();
  synapses.forEach(s => synSet.add(s.pre + "_" + s.post));
  render();
});
btnPain.addEventListener("click", () => {
  PAIN_ON = !PAIN_ON;
  btnPain.classList.toggle("active", PAIN_ON);
  btnPain.textContent = PAIN_ON ? "停止疼痛" : "施加疼痛";
  render();
});
btnSpeed.addEventListener("click", () => {
  const idx = SPEEDS.indexOf(speed);
  speed = SPEEDS[(idx + 1) % SPEEDS.length];
  spdVal.textContent = speed + "×";
  btnSpeed.textContent = speed >= 16 ? "减速" : "加速";
});

render();
</script>
</body>
</html>
"""


# ---------------- 主流程 ----------------

if __name__ == "__main__":
    random.seed(SEED)   # 固定全局随机种子，保证长程连接与终端报告可复现
    neurons, synapses = build_network()

    # 保存初始网络快照（HTML 里的 JS 从初始网络重新跑一遍完整演化）
    init_data = {
        "neurons": [{"id": n.id, "type": n.type, "B": n.B} for n in neurons],
        "synapses": [{"pre": s.pre, "post": s.post, "W": s.W, "gain": s.gain, "signal": s.signal}
                     for s in synapses],
    }

    converged_at, stats, snapshots, pain_compare = run_demo(neurons, synapses)
    report(neurons, synapses, converged_at, stats, snapshots,
           init_syn_count=len(init_data["synapses"]), pain_compare=pain_compare)

    data = {
        "neurons": init_data["neurons"],
        "synapses": init_data["synapses"],
        "nodeColors": NODE_COLORS,
        "nodeNames": NODE_NAMES,
        "signalTypes": SIGNAL_TYPES,
        "injectPeriod": INJECT_PERIOD,
        "lr": LR,
        "pruneW": PRUNE_W,
        "inactiveMax": INACTIVE_MAX,
        "negMax": NEG_STREAK_MAX,
        "pathEvery": PATH_CHECK_EVERY,
        "pathMin": PATH_COUNT_MIN,
        "maxSteps": MAX_STEPS,
        "convWindow": CONV_WINDOW,
        "convEps": CONV_EPS,
        "painId": [n["id"] for n in init_data["neurons"] if n["type"] == "pain"][0],
        "pInhibit": P_INHIBIT,
    }
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open("drn_learning_demo.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n已生成可视化：drn_learning_demo.html（浏览器打开，可交互重跑完整演化）")
