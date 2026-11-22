def rank(scores, Top_N=10):
    return sorted(scores.keys(), key=scores.get, reverse=True)[:Top_N]