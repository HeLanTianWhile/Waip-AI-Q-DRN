# -*- coding: utf-8 -*-
"""追踪 net-3 关键神经元随 tick 的值，定位点击链路断点。"""
import json, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path("data").resolve()))
from drn_core import Network, PIXEL_COUNT

ROOT = Path(r"data\Agent\net-3")
cfg = json.loads((ROOT / "Neural Configuration Database.json").read_text(encoding="utf-8"))
syn = json.loads((ROOT / "Synapses Database.json").read_text(encoding="utf-8"))
net = Network(cfg, syn)
id2idx = net.id_to_idx

incoming = defaultdict(list)
for s in syn.values():
    ct = s["Connection Target"]; a, b = ct[0], ct[1]
    if s["Type"] != "unidirectional synapse": continue
    incoming[a].append((b, float(s["Weight"])))

def bias(nid): return float(cfg[nid].get("bias", 0))
def nid_type(nid): return cfg[nid].get("IP")[0]

# out3
out3 = [n for n, c in cfg.items() if c.get("IP") == ["Output", 3]][0]
# DET: bias ~ -329.5, Normal
det = [n for n, c in cfg.items() if nid_type(n) == "Normal" and abs(bias(n) - (-0.05 * (6640 - 50))) < 1e-6][0]
# NOTH: bias6, 2 inputs
noth = sorted([n for n, c in cfg.items() if nid_type(n) == "Normal" and abs(bias(n) - 6.0) < 1e-9 and len(incoming[n]) == 2], key=int)
# mem_a (EN): bias2, 3 inputs (noth +10, mem_b +6, reset -6)
en = sorted([n for n, c in cfg.items() if nid_type(n) == "Normal" and abs(bias(n) - 2.0) < 1e-9 and len(incoming[n]) == 3], key=int)
# mem_b: bias2, 2 inputs (mem_a +6, reset -6)
memb = sorted([n for n, c in cfg.items() if nid_type(n) == "Normal" and abs(bias(n) - 2.0) < 1e-9 and len(incoming[n]) == 2], key=int)
# ck: bias6, 3 inputs
ck = sorted([n for n, c in cfg.items() if nid_type(n) == "Normal" and abs(bias(n) - 6.0) < 1e-9 and len(incoming[n]) == 3], key=int)
# clk0: first Normal neuron id (inputs then outputs then clk)
normals = sorted([n for n, c in cfg.items() if nid_type(n) == "Normal"], key=int)
clk0 = normals[0]

print(f"out3={out3} DET={det} NOTH={noth} EN(mem_a)={en} mem_b={memb}")
print(f"#ck={len(ck)} first ck={ck[0]} clk0={clk0}")
# sanity: print incoming of EN1 and ck0
for label, n in [("EN1", en[0]), ("CK0", ck[0])]:
    print(f"  {label} {n} incoming:", [(s, w) for s, w in incoming[n]])

def yv(nid):
    v = net.y[id2idx[nid]]
    return None if v != v else round(float(v), 3)

BG = 15
def screen(left=False):
    px = bytearray([BG] * PIXEL_COUNT)
    if left:
        for py in range(40, 130, 3):
            for px_ in range(20, 160, 3):
                px[py * 390 + px_] = 227
    return bytes(px)

net.reset_state()
cx, cy, vy = 195.0, 422.0, 422.0
print("\ntick  out0  out1  out3 | CLK0 DET  NOTH1 EN1  mb1  CK0  | cursor(x,y)")
for t in range(140):
    left = False
    net.inject(screen(left), cx, cy, vy)
    net.propagate()
    o = net.read_outputs()
    cx = max(0.0, min(390.0, cx + 390.0 * o[0]))
    cy = max(0.0, min(844.0, cy + 844.0 * o[1]))
    vy = max(96.0, min(748.0, vy + 844.0 * o[2]))
    if t < 30 or (t >= 108 and t <= 116) or o[3] > 0:
        en1_v = yv(en[0]); mb1_v = yv(memb[0]); ck0_v = yv(ck[0])
        print(f"{t:3d}  {o[0]:+.2f} {o[1]:+.2f} {o[3]:+.2f} | "
              f"{(yv(clk0) or 0):+.1f} {(yv(det) or 0):+.1f} {(yv(noth[0]) or 0):+.1f} "
              f"{(en1_v if en1_v is not None else 0):+.1f} {(mb1_v if mb1_v is not None else 0):+.1f} "
              f"{(ck0_v if ck0_v is not None else 0):+.1f} | ({cx:5.1f},{cy:5.1f})")
