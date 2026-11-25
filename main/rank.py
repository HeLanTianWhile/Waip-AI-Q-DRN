def rank(scores, Top_N=10):
    return sorted(scores.keys(), key=scores.get, reverse=True)[:Top_N]


def rank_unique(scores, Top_N=10, digits=4):
    """按分数从高到低排序，但同分只保留一个，用来打破“前10永远是那10个克隆”的趋同。

    前 Top_N 名里若出现分数相同的模型，只留排名最靠前的那一个，
    空出来的位置自动由后面第一个不同分的模型补上。
    """
    order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    seen = set()
    out = []
    for name, score in order:
        key = round(score, digits)
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= Top_N:
            break
    return out
