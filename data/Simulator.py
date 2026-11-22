# -*- coding: utf-8 -*-
"""
无头评估器：用一张静态 ChatUI 屏幕截图，让单个网络跑 50 tick，
返回轨迹（供回放）和鼠标/点击位置（供评分）。

屏幕渲染只发生一次并缓存到 data/TRAIN/screen.npy；渲染必须在
主线程调用（Qt 限制），之后评估是纯 numpy，可在后台线程跑。
"""

import os
import time
from pathlib import Path

import numpy as np

import drn_core

ROOT_DIR = Path(__file__).resolve().parent.parent
HTML_FILE = ROOT_DIR / "ChatUI" / "index.html"

SCREEN_W, SCREEN_H = drn_core.SCREEN_W, drn_core.SCREEN_H
VIEW_H = drn_core.VIEW_H
TICKS_PER_RUN = 50


def _render_screen_via_qt():
    """离屏渲染 ChatUI 并返回 (844,390) uint8 灰度图。必须在主线程调用。"""
    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWebEngineWidgets import QWebEngineView

    app = QApplication.instance() or QApplication([])
    view = QWebEngineView()
    view.resize(SCREEN_W, SCREEN_H)
    view.show()
    view.load(QUrl.fromLocalFile(str(HTML_FILE)))

    def _wait(ms):
        lp = QEventLoop()
        QTimer.singleShot(ms, lp.quit)
        lp.exec()

    loop = QEventLoop()
    view.loadFinished.connect(loop.quit)
    QTimer.singleShot(20000, loop.quit)
    loop.exec()
    # 等首帧 paint（离屏 grab 拿到的是未初始化缓冲，必须等渲染）
    _wait(400)
    # 隐藏光标/视野/点击环，拿到干净背景；评估时会自己叠加光标
    view.page().runJavaScript(
        "document.getElementById('cursor').style.display='none';"
        "document.getElementById('aperture').style.display='none';"
        "document.getElementById('clickring').style.display='none';"
    )
    _wait(250)

    image = view.grab().toImage()
    if image.isNull():
        return None
    # view 已固定 390x844，直接取灰度
    image = image.convertToFormat(QImage.Format_Grayscale8)
    bits = memoryview(image.constBits())
    stride = image.bytesPerLine()
    arr = np.empty((SCREEN_H, SCREEN_W), dtype=np.uint8)
    for row in range(SCREEN_H):
        arr[row, :] = np.frombuffer(
            bits[row * stride:row * stride + SCREEN_W], dtype=np.uint8
        )
    view.close()
    return arr


def _synthetic_screen():
    """渲染失败时的兜底：深色背景 + W 键位置的高亮方块，保证训练不被阻塞。"""
    arr = np.full((SCREEN_H, SCREEN_W), 18, dtype=np.uint8)
    # W 键实测位置 (43.5, 676) ~ (77, 712)
    arr[676:712, 43:78] = 255
    # 键盘区域其它键给点弱纹理，避免全黑
    arr[676:712, 5:40] = 40
    arr[676:712, 81:115] = 40
    return arr


def load_or_render_screen(cache_path):
    """优先读缓存；不存在则渲染并缓存。返回 (844,390) uint8。"""
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            return np.load(str(cache_path))
        except Exception:
            pass
    arr = _render_screen_via_qt()
    if arr is None or arr.shape != (SCREEN_H, SCREEN_W):
        arr = _synthetic_screen()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), arr)
    except Exception:
        pass
    return arr


def evaluate(cfg, syn, screen, ticks=TICKS_PER_RUN, cursor_overlay=True):
    """
    跑一个网络 ticks 步，返回：
      mouse_positions: [(x,y), ...] 每 tick 的光标位置
      click_positions: [(x,y), ...] 发生点击的位置
      frames: [{cx,cy,vy,click}, ...] 供回放
    """
    net = drn_core.Network(cfg, syn)
    net.reset_state()

    cursor_x, cursor_y = SCREEN_W / 2.0, SCREEN_H / 2.0
    view_y = SCREEN_H / 2.0

    mouse_positions = []
    click_positions = []
    frames = []

    for _ in range(ticks):
        top = int(round(view_y - VIEW_H / 2.0))
        top = drn_core.clamp(top, 0, SCREEN_H - VIEW_H)
        crop = screen[top:top + VIEW_H, :].copy()
        if cursor_overlay:
            rx = int(round(cursor_x))
            ry = int(round(cursor_y - top))
            if 0 <= rx < SCREEN_W and 0 <= ry < VIEW_H:
                y0 = max(0, ry - 2); y1 = min(VIEW_H, ry + 3)
                x0 = max(0, rx - 2); x1 = min(SCREEN_W, rx + 3)
                crop[y0:y1, x0:x1] = 255
        pixels = crop.ravel()

        net.inject(pixels, cursor_x, cursor_y, view_y)
        net.propagate()
        outputs = net.read_outputs()

        cursor_x = drn_core.clamp(cursor_x + SCREEN_W * outputs[0], 0.0, float(SCREEN_W))
        cursor_y = drn_core.clamp(cursor_y + SCREEN_H * outputs[1], 0.0, float(SCREEN_H))
        view_y = drn_core.clamp(
            view_y + SCREEN_H * outputs[2],
            float(drn_core.VIEW_Y_MIN), float(drn_core.VIEW_Y_MAX),
        )
        click = outputs[3] > 0.0

        mouse_positions.append((round(cursor_x, 2), round(cursor_y, 2)))
        if click:
            click_positions.append((round(cursor_x, 2), round(cursor_y, 2)))
        frames.append({
            "cx": round(cursor_x, 2),
            "cy": round(cursor_y, 2),
            "vy": round(view_y, 2),
            "click": bool(click),
        })

    return {
        "mouse_positions": mouse_positions,
        "click_positions": click_positions,
        "frames": frames,
    }
