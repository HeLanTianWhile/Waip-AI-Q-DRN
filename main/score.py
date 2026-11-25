import json
import math
from pathlib import Path

D = math.hypot(390, 844)
VIEW_RANGE = 748 - 96

STEP_DEFAULTS = {
    "mouse_move": {"tol": 20.0, "ticks": 50, "weight": 1.0},
    "click": {"tol": 17.0, "ticks": 50, "weight": 1.0},
    "move_view": {"tol": 40.0, "ticks": 50, "weight": 1.0},
}


def load_lesson(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = []
    for item in raw:
        kind = item["type"]
        if kind not in STEP_DEFAULTS:
            raise ValueError(f"unknown step type: {kind}")
        merged = dict(STEP_DEFAULTS[kind])
        merged.update(item)
        step = {
            "type": kind,
            "tol": float(merged["tol"]),
            "ticks": max(1, int(merged["ticks"])),
            "weight": float(merged["weight"]),
            "y": float(item["y"]),
        }
        if kind in ("mouse_move", "click"):
            step["x"] = float(item["x"])
        steps.append(step)
    if not steps:
        raise ValueError("empty lesson")
    return steps


def _step_error(step, frames, indexes):
    """返回 (best_err, hit_tick, n_clicks, n_wrong)。

    click 步只看真实发生点击的那些 tick 的位置。“该点却没点”直接记 0，
    保证任何一次点击都比不点更好，避免模型学会“放弃点击”。
    """
    kind = step["type"]
    best = math.inf
    hit_tick = None
    if kind == "move_view":
        for i in indexes:
            err = abs(frames[i]["vy"] - step["y"])
            if err < best:
                best = err
            if hit_tick is None and err <= step["tol"]:
                hit_tick = i
        return best, hit_tick, 0, 0

    cursor_errs = []
    click_errs = []
    n_clicks = 0
    n_wrong = 0
    for i in indexes:
        err = math.hypot(frames[i]["cx"] - step["x"], frames[i]["cy"] - step["y"])
        cursor_errs.append((i, err))
        if frames[i]["click"]:
            n_clicks += 1
            click_errs.append((i, err))
            if err <= step["tol"]:
                if hit_tick is None:
                    hit_tick = i
            else:
                n_wrong += 1

    if kind == "mouse_move":
        for i, err in cursor_errs:
            if err < best:
                best = err
            if hit_tick is None and err <= step["tol"]:
                hit_tick = i
        return best, hit_tick, 0, 0

    if click_errs:
        best = min(err for _i, err in click_errs)
    return best, hit_tick, n_clicks, n_wrong


def score_lesson(steps, frames):
    results = []
    start = 0
    weighted = 0.0
    total_weight = 0.0
    prev_done = True
    for step in steps:
        end = min(len(frames), start + step["ticks"])
        reachable = prev_done          # 前一步失败后，这一步还没轮得到
        indexes = range(start, end)
        best, hit_tick, n_clicks, n_wrong = _step_error(step, frames, indexes)
        done = reachable and hit_tick is not None
        prev_done = done
        norm = VIEW_RANGE if step["type"] == "move_view" else D
        progress = 0.0 if not math.isfinite(best) else max(0.0, 1.0 - best / norm)
        value = 1.0 if done else 0.9 * progress
        if step["type"] == "click":
            if n_clicks == 0:
                # 该点击却没点：0 分。任何一次点击都严格优于不点，避免模型放弃点击。
                value = 0.0
            elif not done and n_wrong:
                # 点偏了也仍然给分（只是略有折扣），绝不低于“不点”
                value *= 1.0 / (1.0 + 0.05 * n_wrong)
        if not reachable:
            # 前序步骤没完成：只保留 1/4 的引导信号，避免“跳步”白拿高分，同时仍给进化留梯度
            value *= 0.25
        weighted += value * step["weight"]
        total_weight += step["weight"]
        results.append({
            "type": step["type"],
            "done": done,
            "tick": hit_tick,
            "error": None if not math.isfinite(best) else round(best, 3),
            "clicks": n_clicks,
            "wrong_clicks": n_wrong,
            "score": round(value, 6),
        })
        start = end

    return {
        "score": weighted / total_weight if total_weight else 0.0,
        "steps": results,
        "completed": all(r["done"] for r in results),
    }


DEFAULT_LESSON = Path(__file__).resolve().parent.parent / "data" / "class" / "class-1.json"
_default_lesson_cache = None


def default_lesson():
    global _default_lesson_cache
    if _default_lesson_cache is None:
        _default_lesson_cache = load_lesson(DEFAULT_LESSON)
    return _default_lesson_cache


def score(mouse_positions, click_positions, frames=None, lesson=None):
    if lesson is None:
        lesson = default_lesson()
    if frames is None:
        clicked = set(click_positions)
        frames = [
            {"cx": x, "cy": y, "vy": 0.0, "click": (x, y) in clicked}
            for x, y in mouse_positions
        ]
    return score_lesson(lesson, frames)["score"]
