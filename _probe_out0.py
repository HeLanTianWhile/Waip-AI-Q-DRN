# -*- coding: utf-8 -*-
"""打印 out0 在每个 tick 的入边贡献，定位 t=10 的瞬态 -0.021 来自哪条突触。"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("data").resolve()))
from drn_core import Network, PIXEL_COUNT

ROOT = Path(r"data\Agent\net-3")
cfg = json.loads((ROOT / "Neural Configuration Database.json").read_text(encoding="utf-8"))
syn = json.loads((ROOT / "Synapses Database.json").read_text(encoding="utf-8"))
net = Network(cfg, syn)

out0 = net.output_idx[0]
curs_x = net.input_idx[PIXEL_COUNT + 2]
curs_y = net.input_idx[PIXEL_COUNT + 3]

m0 = net.tgt == out0
m1 = net.tgt == net.output_idx[1]
srcs0, ws0 = net.src[m0], net.w[m0]
srcs1, ws1 = net.src[m1], net.w[m1]
print(f"out0 has {len(srcs0)} incoming edges; out1 has {len(srcs1)}")

inv = {i: n for n, i in net.id_to_idx.items()}
def bias_of(idx): return float(net.bias[idx])
def name(idx):
    n = inv.get(int(idx))
    if n is None: return f"idx{idx}"
    ip = cfg[n].get("IP")
    return f"{n}({ip[0]}{ip[1] if ip[0]!='Normal' else ''},b={bias_of(idx):g})"

BG = 15
def screen(): return bytes([BG] * PIXEL_COUNT)

net.reset_state()
cx, cy, vy = 195.0, 422.0, 422.0
for t in range(0, 20):
    net.inject(screen(), cx, cy, vy)
    net.propagate()
    o = net.read_outputs()
    yv = net.y
    contrib = ws0 * np.where(np.isnan(yv[srcs0]), 0.0, yv[srcs0])
    order = np.argsort(-np.abs(contrib))[:6]
    parts = []
    for k in order:
        if abs(contrib[k]) < 1e-9: continue
        parts.append(f"{name(srcs0[k])} y={yv[srcs0[k]]:+.3f} w={ws0[k]:+.4f} -> {contrib[k]:+.4f}")
    cn = yv[curs_x]
    print(f"t={t:2d} out0={o[0]:+.4f} out1={o[1]:+.4f} curs=({cx:6.1f},{cy:6.1f}) cxnorm={cn}")
    for p in parts:
        print("      ", p)
    cx = max(0.0, min(390.0, cx + 390.0 * o[0]))
    cy = max(0.0, min(844.0, cy + 844.0 * o[1]))
    vy = max(96.0, min(748.0, vy + 844.0 * o[2]))
