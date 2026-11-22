def rank(scores, Top_N=10):
    return [k for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)][:Top_N]
