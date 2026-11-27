# -*- coding: utf-8 -*-
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"data\Agent\net-3")
cfg = json.loads((ROOT / "Neural Configuration Database.json").read_text(encoding="utf-8"))
syn = json.loads((ROOT / "Synapses Database.json").read_text(encoding="utf-8"))

incoming = defaultdict(list)
outgoing = defaultdict(list)
for sid, s in syn.items():
    ct = s.get("Connection Target")
    if not isinstance(ct, (list, tuple)) or len(ct) != 2: continue
    a, b = ct[0], ct[1]
    if a == b: continue
    w = float(s.get("Weight", 0.0))
    bi = s.get("Type", "bidirectional synapse") != "unidirectional synapse"
    if bi:
        incoming[b].append((a, w)); outgoing[a].append((b, w))
    incoming[a].append((b, w)); outgoing[b].append((a, w))

# special neurons: bias 2.0 (clock), and the few bias 0/6/-324
clock = [nid for nid,c in cfg.items() if abs(float(c.get("bias",0))-2.0)<1e-9 and c.get("IP",[None])[0]=="Normal"]
print(f"clock-ish (bias 2.0, Normal): {len(clock)}")
clock_sorted = sorted(clock, key=int)
print("ids:", clock_sorted[:10], "...", clock_sorted[-5:])

def info(nid):
    c = cfg.get(nid, {})
    return f"{nid}(b={c.get('bias')})"

# For each clock neuron, print outgoing (to find chain) — only strong weights
print("\n=== clock chain outgoing (|w|>=4) ===")
for nid in clock_sorted:
    outs = [(t,w) for t,w in outgoing.get(nid,[]) if abs(w)>=4]
    if outs:
        print(f"  {info(nid)} -> " + ", ".join(f"{info(t)}[{w}]" for t,w in outs))

# detector + gates: bias -324, 0, 6
print("\n=== special (bias in {-324,0,6}) Normal ===")
for nid,c in sorted(cfg.items(), key=lambda x:int(x[0])):
    b = float(c.get("bias",0))
    if c.get("IP",[None])[0]!="Normal": continue
    if b in (-324.0, 0.0, 6.0):
        ins = [(s,w) for s,w in incoming.get(nid,[])]
        outs = [(t,w) for t,w in outgoing.get(nid,[])]
        print(f"\n  {info(nid)}")
        print(f"    in : " + ", ".join(f"{info(s)}[{w}]" for s,w in sorted(ins,key=lambda x:-abs(x[1]))[:8]))
        print(f"    out: " + ", ".join(f"{info(t)}[{w}]" for t,w in sorted(outs,key=lambda x:-abs(x[1]))[:8]))
