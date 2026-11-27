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


def _step_stats(step, frames, indexes):
    kind = step["type"]
    best = math.inf
    hit_tick = None
    hit_ticks = 0
    n_actions = 0
    n_clicks = 0
    n_wrong = 0
    for i in indexes:
        frame = frames[i]
        previous = frames[i - 1] if i > 0 else None
        if kind == "move_view":
            acted = previous is not None and frame["vy"] != previous["vy"]
            err = abs(frame["vy"] - step["y"])
        elif kind == "mouse_move":
            acted = previous is not None and (
                frame["cx"] != previous["cx"] or frame["cy"] != previous["cy"]
            )
            err = math.hypot(frame["cx"] - step["x"], frame["cy"] - step["y"])
        else:
            acted = bool(frame["click"])
            err = math.hypot(frame["cx"] - step["x"], frame["cy"] - step["y"])
            if acted:
                n_clicks += 1
        if not acted:
            continue
        n_actions += 1
        if err < best:
            best = err
        if err <= step["tol"]:
            hit_ticks += 1
            if hit_tick is None:
                hit_tick = i
        elif kind == "click":
            n_wrong += 1
    return best, hit_tick, hit_ticks, n_actions, n_clicks, n_wrong


def score_lesson(steps, frames):
    results = []
    start = 0
    total = 0.0
    total_weight = 0.0
    for step in steps:
        end = min(len(frames), start + step["ticks"])
        window = max(0, end - start)
        best, hit_tick, hit_ticks, n_actions, n_clicks, n_wrong = _step_stats(
            step, frames, range(start, end)
        )
        norm = VIEW_RANGE if step["type"] == "move_view" else D
        if math.isfinite(best):
            excess = max(0.0, best - step["tol"])
            progress = max(0.0, 1.0 - excess / norm)
        else:
            progress = 0.0
        value = step["weight"] * progress
        total += value
        total_weight += step["weight"]
        results.append({
            "type": step["type"],
            "done": hit_ticks > 0,
            "tick": hit_tick,
            "ticks": window,
            "hit_ticks": hit_ticks,
            "actions": n_actions,
            "error": None if not math.isfinite(best) else round(best, 3),
            "clicks": n_clicks,
            "wrong_clicks": n_wrong,
            "score": round(value, 6),
        })
        start = end

    return {
        "score": total,
        "max_score": total_weight,
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
