import math, random, copy
import numpy as np

P_ADD_SYN = 0.08
P_DEL_SYN = 0.03
P_ADD_NEU = 0.08
P_DEL_NEU = 0.03


def _nid(i):
    return f"{i:04d}"


def _split_roles(cfg):
    hidden, io = [], []
    for k, v in cfg.items():
        r = v["IP"][0]
        if r in ("Memory", "Normal"):
            hidden.append(k)
        elif r in ("Input", "Output"):
            io.append(k)
    return hidden, io


def mutation(cfg_parent, syn_parent, mutation_rate=0.001):
    cfg = copy.deepcopy(cfg_parent)
    syn = copy.deepcopy(syn_parent)
    hidden_ids, io_ids = _split_roles(cfg_parent)

    cat = {"synapse": False, "hidden": False, "io": False}
    cat[random.choice(("synapse", "hidden", "io"))] = True

    if cat["synapse"]:
        ids = list(syn)
        n = min(max(1, round(len(ids) * mutation_rate)), len(ids))
        idx = np.random.choice(len(ids), n, replace=False)
        deltas = np.random.uniform(-1.0, 1.0, n)
        for j, i in enumerate(idx):
            sid = ids[i]
            syn[sid]["Weight"] = math.tanh(syn[sid]["Weight"] + float(deltas[j]))

        if random.random() < P_ADD_SYN:
            nxt = max(int(k) for k in syn) + 1
            for _ in range(random.randint(1, 3)):
                a = random.choice(hidden_ids + io_ids)
                b = random.choice(hidden_ids)
                if a == b:
                    continue
                syn[_nid(nxt)] = {"Weight": random.uniform(-1.0, 1.0),
                                  "Connection Target": [a, b]}
                nxt += 1

        if random.random() < P_DEL_SYN:
            for sid in random.sample(ids, min(3, len(ids))):
                syn.pop(sid, None)

    if cat["hidden"]:
        n = min(max(1, round(len(hidden_ids) * mutation_rate)), len(hidden_ids))
        picked = random.sample(hidden_ids, n)
        for nid in picked:
            cfg[nid]["bias"] = math.tanh(cfg[nid]["bias"] + random.uniform(-1.0, 1.0))

        if random.random() < P_ADD_NEU:
            nxt = max(int(k) for k in cfg) + 1
            for _ in range(random.randint(1, 2)):
                new = _nid(nxt)
                cfg[new] = {"bias": random.uniform(-1.0, 1.0), "IP": ["Normal", nxt]}
                nxt += 1
                nxt_s = max(int(k) for k in syn) + 1
                for _ in range(random.randint(1, 5)):
                    a = random.choice(hidden_ids + io_ids)
                    syn[_nid(nxt_s)] = {"Weight": random.uniform(-1.0, 1.0),
                                        "Connection Target": [a, new]}
                    nxt_s += 1

        if random.random() < P_DEL_NEU:
            for nid in random.sample(hidden_ids, min(2, len(hidden_ids))):
                cfg.pop(nid, None)
                dead = [sid for sid, v in syn.items() if nid in v["Connection Target"]]
                for sid in dead:
                    syn.pop(sid, None)

    if cat["io"]:
        n = min(max(1, round(len(io_ids) * mutation_rate)), len(io_ids))
        for nid in random.sample(io_ids, n):
            cfg[nid]["bias"] = math.tanh(cfg[nid]["bias"] + random.uniform(-1.0, 1.0))

    return cfg, syn
