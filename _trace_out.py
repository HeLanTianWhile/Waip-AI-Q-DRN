# -*- coding: utf-8 -*-
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"data\Agent\net-3")
cfg = json.loads((ROOT / "Neural Configuration Database.json").read_text(encoding="utf-8"))
syn = json.loads((ROOT / "Synapses Database.json").read_text(encoding="utf-8"))

# incoming map
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

def info(nid):
    c = cfg.get(nid, {})
    return f"{nid}(IP={c.get('IP')} bias={c.get('bias')})"

for out_id in ["74885","74886","74887","74888"]:
    print(f"\n=== OUTPUT {out_id} {info(out_id)} ===")
    ins = incoming.get(out_id, [])
    print(f"  incoming ({len(ins)}):")
    for src, w in sorted(ins, key=lambda x: -abs(x[1])):
        print(f"    {info(src)}  --w={w}-->  {out_id}")

# Also: what are the distinct bias values among normal neurons? (to identify clock/phase/detector)
from collections import Counter
bias_hist = Counter()
ip_hist = Counter()
for nid, c in cfg.items():
    ip = c.get("IP")
    if isinstance(ip,(list,tuple)) and len(ip)>=2:
        ip_hist[(ip[0],)] += 1
    bias_hist[round(float(c.get("bias",0.0)),4)] += 1
print("\n=== bias histogram (non-2.0) ===")
for b, cnt in sorted(bias_hist.items()):
    if b != 2.0:
        print(f"  bias={b}: {cnt} neurons")
print("=== IP kind histogram ===")
for k, cnt in ip_hist.items():
    print(f"  {k}: {cnt}")
