# -*- coding: utf-8 -*-
import json
from pathlib import Path

ROOT = Path(r"data\Agent\net-3")
cfg = json.loads((ROOT / "Neural Configuration Database.json").read_text(encoding="utf-8"))
syn = json.loads((ROOT / "Synapses Database.json").read_text(encoding="utf-8"))
st  = json.loads((ROOT / "Neural State Database.json").read_text(encoding="utf-8"))

print("neurons:", len(cfg), "synapses:", len(syn))

# categorize
inputs = []
outputs = []
others = []
for nid, c in cfg.items():
    ip = c.get("IP")
    if isinstance(ip, (list, tuple)) and len(ip) >= 2 and isinstance(ip[1], int):
        if ip[0] == "Input":
            inputs.append((nid, ip[1]))
        elif ip[0] == "Output":
            outputs.append((nid, ip[1], c.get("bias")))
        else:
            others.append((nid, ip, c.get("bias")))
    else:
        others.append((nid, ip, c.get("bias")))

print("inputs:", len(inputs), "outputs:", len(outputs), "others:", len(others))
print("--- outputs ---")
for nid, idx, b in sorted(outputs, key=lambda x: x[1]):
    print(f"  out[{idx}] = {nid}  bias={b}")

print("--- non-input neurons (id, IP, bias) ---")
for nid, ip, b in sorted(others, key=lambda x: int(x[0])):
    print(f"  {nid}: IP={ip} bias={b}")
