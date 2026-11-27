# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("data").resolve()))
from drn_core import Network, PIXEL_COUNT

ROOT = Path(r"data\Agent\net-3")
cfg = json.loads((ROOT / "Neural Configuration Database.json").read_text(encoding="utf-8"))
syn = json.loads((ROOT / "Synapses Database.json").read_text(encoding="utf-8"))
net = Network(cfg, syn)
id2idx = net.id_to_idx

# 找到关键神经元：输出、DET、NOTH1、EN1、CHECK1、第一个 CK
def find(syn_predicate):
    pass
# 直接按 bias 特征找
out3 = [nid for nid,c in cfg.items() if c.get("IP")==["Output",3]][0]
# DET bias ~ -328
det = [nid for nid,c in cfg.items() if abs(float(c.get("bias",0)) - (-0.05*(6640-50)))<1e-6 and c.get("IP")[0]=="Normal"][0]
print("out3",out3,"DET",det,"bias",cfg[det]["bias"])

# 找 NOTH1: bias 6, incoming from DET(-6) and check(+6)
# 找 EN1: bias 2, self-loop
# 简单点：追踪 out3 的所有上游
from collections import defaultdict
incoming=defaultdict(list)
for sid,s in syn.items():
    ct=s["Connection Target"]; a,b=ct[0],ct[1]
    if s["Type"]!="unidirectional synapse": continue
    incoming[a].append((b,float(s["Weight"])))
def info(nid):
    c=cfg.get(nid,{}); return f"{nid}(b={c.get('bias')})"
print("\nout3 incoming:")
for s,w in incoming.get(out3,[]):
    print("  ",info(s),w)
# 找其中一个 CK（bias6），看它的上游
cks=[nid for nid,c in cfg.items() if c.get("IP")[0]=="Normal" and abs(float(c.get("bias"))-6.0)<1e-9]
print(f"\nbias6 neurons (CK candidates): {len(cks)}")
# 第一个 CK 的上游
ck0=sorted(cks,key=int)[0]
print("CK0",info(ck0))
for s,w in incoming.get(ck0,[]):
    print("  in :",info(s),w)

# NOTH / EN 查找：bias2 有自环的
en=[nid for nid,c in cfg.items() if c.get("IP")[0]=="Normal" and abs(float(c.get("bias"))-2.0)<1e-9 and any(x[0]==nid for x in incoming.get(nid,[]))]
print(f"\nbias2 self-loop (EN): {len(en)} -> {sorted(en,key=int)[:4]}")
noth=[nid for nid,c in cfg.items() if c.get("IP")[0]=="Normal" and abs(float(c.get("bias"))-6.0)<1e-9 and len(incoming.get(nid,[]))==2]
print(f"bias6 with 2 inputs (NOTH): {sorted(noth,key=int)[:4]}")

# 运行仿真打印
def yv(nid):
    v=net.y[id2idx[nid]]; return None if v!=v else round(float(v),3)
BG=15
def screen():
    return bytes([BG]*PIXEL_COUNT)
net.reset_state()
cx,cy,vy=195.0,422.0,422.0
print("\ntick  outX outY outVY out3 | CLK0 DET NOTH1 EN1 CK0")
# 取第一个 NOTH 和 EN
noth1=sorted(noth,key=int)[0]
en1=sorted(en,key=int)[0]
clk0=None
for nid,c in cfg.items():
    if c.get("IP")[0]=="Normal" and abs(float(c.get("bias"))-2.0)<1e-9 and not any(x[0]==nid for x in incoming.get(nid,[])):
        # clock neurons: bias2, has incoming from view_x or previous clk
        pass
# CLK0 = first normal after outputs = 74889
clk0="74889"
for t in range(40):
    net.inject(screen(), cx, cy, vy)
    net.propagate()
    o=net.read_outputs()
    cx=max(0,min(390,cx+390*o[0])); cy=max(0,min(844,cy+844*o[1])); vy=max(96,min(748,vy+844*o[2]))
    if t<30 or o[3]>0:
        print(f"{t:3d}  {o[0]:+.2f} {o[1]:+.2f} {o[2]:+.2f} {o[3]:+.2f} | "
              f"{(yv(clk0) or 0):+.1f} {(yv(det) or 0):+.1f} {(yv(noth1) or 0):+.1f} {(yv(en1) or 0):+.1f} {(yv(ck0) or 0):+.1f}")
