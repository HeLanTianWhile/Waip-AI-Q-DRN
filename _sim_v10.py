# -*- coding: utf-8 -*-
"""仿真 v10 net-3，记录时钟链与输出的真实时序。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("data").resolve()))
from drn_core import Network, SCREEN_W, SCREEN_H, VIEW_W, VIEW_H, VIEW_X, PIXEL_COUNT

ROOT = Path(r"data\Agent\net-3")
cfg = json.loads((ROOT / "Neural Configuration Database.json").read_text(encoding="utf-8"))
syn = json.loads((ROOT / "Synapses Database.json").read_text(encoding="utf-8"))
net = Network(cfg, syn)
print("N =", net.N)

# 全黑屏幕（空聊天），视野在顶部 y=96，光标中心
def blank():
    return bytes([0]*PIXEL_COUNT)

# 模拟闭环：每个 tick 注入 + 传播，记录输出与关键神经元
net.reset_state()
clk_ids = [str(74889+i) for i in range(82)]  # clock chain
key_ids = {
    "CLK0":"74889","PCHECK1":"74971","PSEND1":"74972","PCHECK2":"74973","PSEND2":"74974","PDONE":"74975",
    "DET":"81536","NOTH1":"81537","NOTH2":"81538","SEL1":"81539","SEL2":"81540",
    "OUTX":"74885","OUTY":"74886","OUTVY":"74887","OUTCLICK":"74888",
}
# map id -> idx
id2idx = net.id_to_idx
def yval(nid):
    v = net.y[id2idx[nid]]
    return None if v != v else round(float(v),3)

cursor_x, cursor_y, view_y = 195.0, 422.0, 422.0
print("tick  view    cursor      outX  outY  outVY outClk | CLK0 PCHK1 PSND1 PCHK2 PSND2 PDONE DET NOTH1 SEL1")
for t in range(90):
    px = blank()
    net.inject(px, cursor_x, cursor_y, view_y)
    net.propagate()
    outs = net.read_outputs()
    cursor_x = max(0,min(390, cursor_x + 390*outs[0]))
    cursor_y = max(0,min(844, cursor_y + 844*outs[1]))
    view_y   = max(96,min(748, view_y + 844*outs[2]))
    click = "CLICK" if outs[3]>0 else ""
    if t < 70 or click:
        print(f"{t:3d}  {view_y:6.1f} ({cursor_x:5.1f},{cursor_y:5.1f}) "
              f"{outs[0]:+.2f} {outs[1]:+.2f} {outs[2]:+.2f} {outs[3]:+.2f} {click:5} | "
              + " ".join(f"{(yval(key_ids[k]) or 0):+.1f}" for k in ["CLK0","PCHECK1","PSEND1","PCHECK2","PSEND2","PDONE","DET","NOTH1","SEL1"]))
