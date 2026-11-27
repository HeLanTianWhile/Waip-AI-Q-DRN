# -*- coding: utf-8 -*-
"""
生成 net-3（v11）：真正通过逐键点击发消息。

架构：
  - 输入 74884（74880 像素 + view_x/view_y/cursor_x/cursor_y）
  - 时钟链（上升填充）CLK[0..T-1]，CLK[0] 由 view_x 常驱动，每级 +8、bias2
  - 相位窗 = tanh(6*CLK[a] - 6*CLK[b])，相邻窗无缝交接
  - 视觉检测器：左侧采样区逐像素二值化 -> DET -> NOTH（检测到左侧有消息则抑制）
  - EN1/EN2 锁存：CHECK 时若无消息则点亮，直到消息结束才复位
  - 光标伺服：out0/out1 = tanh(K*(Σ PH_k*(x_k/W) - cursor_norm))，逐键切换目标
  - 点击：每键一个 1-tick 脉冲 CK_k = tanh(6*CLK[a]-6*CLK[b] + 6*EN - 6) -> out3
  - 视野：out2 恒伺服到 96（看聊天顶部）
发消息必须由网络逐键点击完成，Python 端不再有任何文本注入。
"""
import json
from pathlib import Path

OUT_DIR = Path(r"data\Agent\net-3")
SCREEN_W, SCREEN_H = 390, 844
VIEW_W, VIEW_H = 390, 192
PIXEL_COUNT = VIEW_W * VIEW_H        # 74880
INPUT_COUNT = PIXEL_COUNT + 4        # 74884

# ---- 按键中心坐标（来自 ChatUI 离屏测量）----
KEYS = {
    'q': (22, 694), 'w': (60, 694), 'e': (99, 694), 'r': (137, 694), 't': (176, 694),
    'y': (214, 694), 'u': (253, 694), 'i': (291, 694), 'o': (330, 694), 'p': (368, 694),
    'a': (24, 735), 's': (67, 735), 'd': (109, 735), 'f': (152, 735), 'g': (195, 735),
    'h': (238, 735), 'j': (281, 735), 'k': (323, 735), 'l': (366, 735),
    'z': (67, 776), 'x': (109, 776), 'c': (152, 776), 'v': (195, 776), 'b': (238, 776),
    'n': (281, 776), 'm': (323, 776),
    '，': (32, 817), '空格': (141, 817), '？': (250, 817), 'Enter': (333, 817),
    'send': (355, 646),
}
PARK = (166, 646)   # 等待/结束时光标停放点（输入框，无点击）

MSG1 = ['n', 'i', '空格', 'h', 'a', 'o', '空格', '，',
        'n', 'i', '空格', 'z', 'a', 'i', '空格', 'm', 'a', '空格', '？', 'send']
MSG2 = ['n', 'i', '空格', 'd', 'a', 'o', '空格', 'd', 'i', '空格',
        'z', 'a', 'i', '空格', 'b', 'u', '空格', 'z', 'a', 'i', '空格', '？', 'send']

GAIN = 1.0          # 光标伺服增益
FIRST_MOVE = 6      # 每条消息首键的移动 tick 数（光标从远处/停放点过来）
MOVE = 4            # 普通键的移动 tick 数
# 点击管线：win -> ck_and -> ck_out -> out3 共 3 tick 延迟。
# 因此每键槽长 = 移动 tick + 3，把点击窗放在槽末前 3 tick，out3 正好在槽末（光标已停在键上）触发。
L_FIRST = FIRST_MOVE + 3
L_KEY = MOVE + 3

# ---- 时间槽 ----
def slot(kind, target, length, en):
    return {"kind": kind, "target": target, "length": length, "en": en}

slots = []
slots.append(slot("idle", KEYS[MSG1[0]], 8, None))     # setup：把光标从中心移向首键
for i, k in enumerate(MSG1):
    slots.append(slot("key", KEYS[k], L_FIRST if i == 0 else L_KEY, "EN1"))
slots.append(slot("idle", PARK, 22, None))             # 等待 + 第二次检测
for i, k in enumerate(MSG2):
    slots.append(slot("key", KEYS[k], L_FIRST if i == 0 else L_KEY, "EN2"))
slots.append(slot("idle", PARK, 12, None))             # 结束

# 分配时钟下标
pos = 0
for s in slots:
    s["start"] = pos
    pos += s["length"]
    s["end"] = pos
T = pos + 3   # 末尾余量

# check 窗口（落在 setup / wait 内）
idle_idxs = [i for i, s in enumerate(slots) if s["kind"] == "idle"]
WAIT_START = slots[idle_idxs[1]]["start"]
CHECK1 = (2, 10)                                        # active t3..t10
CHECK2 = (WAIT_START + 6, WAIT_START + 16)             # 等待末段检测

config = {}
synapses = {}
_syn = [0]
def add_syn(target, source, weight):
    _syn[0] += 1
    synapses[f"{_syn[0]:04d}"] = {
        "Weight": float(weight),
        "Connection Target": [target, source],
        "Type": "unidirectional synapse",
    }

_nid = [0]
def new_id():
    _nid[0] += 1
    return f"{_nid[0]:04d}"
def normal(bias=0.0):
    nid = new_id()
    config[nid] = {"IP": ["Normal", 0], "bias": float(bias)}
    return nid

# 输入
input_id = []
for i in range(INPUT_COUNT):
    nid = new_id()
    input_id.append(nid)
    config[nid] = {"IP": ["Input", i], "bias": 0.0}
# 输出
out_id = []
for i in range(4):
    nid = new_id()
    out_id.append(nid)
    config[nid] = {"IP": ["Output", i], "bias": 0.0}

VIEW_X_N = input_id[74880]
VIEW_Y_N = input_id[74881]
CURS_X_N = input_id[74882]
CURS_Y_N = input_id[74883]

# ---- 时钟链 ----
# 重要：CLK 必须全部饱和到 1.0。若 CLK[0] 只有 tanh(10*0.5-2)=0.995 而链上其余级为
# tanh(6)=0.99999，则 win(0, k) = tanh(6*CLK[0] - 6*CLK[k]) ≈ -0.03 会永久残留，
# 经相位权重的稳态偏移把光标恒定拖走 (-8.3,-23.3)px，导致每次点击都落在邻键。
# 用 20 的权重使各级都饱和到 1.0（tanh(18)=1.0），残差 ~1e-6。
clk = [normal(bias=2.0) for _ in range(T)]
add_syn(clk[0], VIEW_X_N, 20.0)
for i in range(T - 1):
    add_syn(clk[i + 1], clk[i], 20.0)

def win(a, b, extra_bias=0.0):
    """相位窗 tanh(6*CLK[a] - 6*CLK[b] - bias). 返回神经元 id。"""
    n = normal(bias=extra_bias)
    add_syn(n, clk[a], 6.0)
    add_syn(n, clk[b], -6.0)
    return n

# ---- 视觉检测器（左侧 x8..172, y31..190）----
PIX = []
for py in range(31, 191, 2):
    for px in range(8, 173, 2):
        pix = normal(bias=5.0)
        add_syn(pix, input_id[py * 390 + px], 10.0)
        PIX.append(pix)
NPIX = len(PIX)
DET = normal(bias=-0.05 * (NPIX - 50))   # 阈值：约 25 个亮像素翻转
for pix in PIX:
    add_syn(DET, pix, 0.05)

# ---- 检测窗 -> NOTH -> EN 锁存 ----
# 注意：drn_core 丢弃 a==b 的自突触，不能用自环做锁存。改用交叉耦合的两个神经元。
# 状态：mem_a=+1,mem_b=+1 锁存；复位脉冲同时抑制两者（1-tick 延迟下单边抑制会被 mem_b 重新拉起）。
def make_check_en(c1, c2):
    check = win(c1, c2)
    noth = normal(bias=6.0)
    add_syn(noth, check, 6.0)
    add_syn(noth, DET, -6.0)
    mem_a = normal(bias=2.0)      # EN 门控信号
    mem_b = normal(bias=2.0)      # 配对保持神经元
    add_syn(mem_a, noth, 10.0)    # 窗口内无消息时置位：10 - 6(mem_b初态-1) - 2 = +2 → 翻正
    add_syn(mem_a, mem_b, 6.0)    # 交叉兴奋：窗口关闭后靠 mem_b=+1 保持（6 - 2 = +4）
    add_syn(mem_b, mem_a, 6.0)    # 反向交叉
    reset = normal(bias=0.0)      # 占位，稍后接复位脉冲
    add_syn(mem_a, reset, -6.0)   # 复位同时清零两侧，避免 1-tick 延迟下重锁存
    add_syn(mem_b, reset, -6.0)
    return check, noth, mem_a, reset

CHECK1_N, NOTH1, EN1, RESET1 = make_check_en(*CHECK1)
CHECK2_N, NOTH2, EN2, RESET2 = make_check_en(*CHECK2)

# ---- 光标伺服输出 ----
# out0 = tanh(GAIN*(Σ PH*(x/W)) - GAIN*cursor_x_norm)  bias 0
# out1 = tanh(GAIN*(Σ PH*(y/H)) - GAIN*cursor_y_norm)  bias 0
add_syn(out_id[0], CURS_X_N, -GAIN)
add_syn(out_id[1], CURS_Y_N, -GAIN)

# 视野恒伺服到 96（复用 v10 标定）
add_syn(out_id[2], VIEW_X_N, 2.0)
add_syn(out_id[2], VIEW_Y_N, -2.0)
config[out_id[2]]["bias"] = 0.7725

# 点击输出 bias 2：任一 CK_k=+1 时触发
config[out_id[3]]["bias"] = 2.0

en_for = {"EN1": EN1, "EN2": EN2}
click_count = 0
for s in slots:
    a, b = s["start"], s["end"]
    ph = win(a, b)                      # 移动窗（整段激活）
    cx, cy = s["target"]
    add_syn(out_id[0], ph, GAIN * (cx / SCREEN_W))
    add_syn(out_id[1], ph, GAIN * (cy / SCREEN_H))
    if s["kind"] == "key":
        # 两层 AND 门：只有 (窗口脉冲 AND EN=on) 才产生 +1，其余态输出≈0（不污染 out3 求和）。
        #   ck_and = tanh(8*win + 8*en - 14)：(1,1)->+0.96，其余≈-1
        #   ck_out = tanh(8*ck_and + 8)：ck_and=-1 -> 0，+0.96 -> +1
        # 点击窗脉冲放在 b-4：管线 win->ck_and->ck_out->out3 延迟 3 tick，
        # out3 在 b-1 触发（仍在槽内，光标已停稳，避免跨到下一槽）。
        wp = b - 4
        win_click = win(wp - 1, wp)             # 在 tick wp 输出 +1（1-tick 脉冲）
        ck_and = normal(bias=14.0)
        add_syn(ck_and, win_click, 8.0)
        add_syn(ck_and, en_for[s["en"]], 8.0)
        ck_out = normal(bias=-8.0)
        add_syn(ck_out, ck_and, 8.0)
        add_syn(out_id[3], ck_out, 8.0)
        click_count += 1

# 复位脉冲：MSG1 结束时清 EN1；全部结束时清 EN2
# 在 MSG1 最后一个 key 之后、wait 之前打一个 1-tick 脉冲
msg1_keys = [s for s in slots if s["en"] == "EN1"]
reset1_t = msg1_keys[-1]["end"]
print(f"DBG WAIT_START={WAIT_START} CHECK2={CHECK2} msg1_last_end={msg1_keys[-1]['end']} reset1_t={reset1_t}")
r1p = win(reset1_t, reset1_t + 1, extra_bias=0.0)
add_syn(RESET1, r1p, 8.0)
msg2_keys = [s for s in slots if s["en"] == "EN2"]
reset2_t = msg2_keys[-1]["end"]
r2p = win(reset2_t, reset2_t + 1, extra_bias=0.0)
add_syn(RESET2, r2p, 8.0)

# ---- 写盘 ----
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "Neural Configuration Database.json").write_text(
    json.dumps(config, ensure_ascii=False), encoding="utf-8")
(OUT_DIR / "Synapses Database.json").write_text(
    json.dumps(synapses, ensure_ascii=False), encoding="utf-8")
state = {nid: {"y": "NOT Activate"} for nid in config}
(OUT_DIR / "Neural State Database.json").write_text(
    json.dumps(state, ensure_ascii=False), encoding="utf-8")

print(f"neurons={len(config)}  synapses={len(synapses)}  clock={T}  pix={NPIX}  clicks={click_count}")
print(f"MSG1={len(MSG1)} keys  MSG2={len(MSG2)} keys  total ticks={pos}")

# 自检：能否被引擎编译
import sys
sys.path.insert(0, str(Path("data").resolve()))
from drn_core import Network
net = Network(config, synapses)
print("compiled N =", net.N, "output_idx =", net.output_idx.tolist())
