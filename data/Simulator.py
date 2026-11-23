# -*- coding: utf-8 -*-
"""
无头评估器：用一张静态 ChatUI 屏幕截图，让单个网络跑 50 tick，
返回轨迹（供回放）和鼠标/点击位置（供评分）。

屏幕渲染只发生一次并缓存到 data/TRAIN/screen.npy；光标精灵同样只渲染一次，
缓存到 data/TRAIN/cursor.npz。渲染必须在主线程调用（Qt 限制），之后评估是
纯 numpy，可在后台线程跑。
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

CURSOR_SIZE = 20          # 与 ChatUI 中 #cursor 的 20x20 一致
# ChatUI 光标 SVG 的轮廓点（viewBox 0 0 24 24）
_CURSOR_PATH = [
    (4.0, 2.0), (4.0, 20.0), (9.2, 15.2), (12.4, 22.0),
    (15.2, 20.6), (12.0, 14.0), (19.0, 13.6),
]


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


def _render_cursor_via_qt():
    """按 ChatUI 的光标 SVG 渲染 20x20 精灵，返回 (灰度 float32, alpha float32)。"""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

    img = QImage(CURSOR_SIZE, CURSOR_SIZE, QImage.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.scale(CURSOR_SIZE / 24.0, CURSOR_SIZE / 24.0)

    path = QPainterPath()
    path.moveTo(*_CURSOR_PATH[0])
    for pt in _CURSOR_PATH[1:]:
        path.lineTo(*pt)
    path.closeSubpath()

    p.fillPath(path, QColor("#e8e8e8"))
    pen = QPen(QColor("#0b0b0b"))
    pen.setWidthF(1.4)
    pen.setJoinStyle(Qt.MiterJoin)
    p.strokePath(path, pen)

    ring_pen = QPen(QColor(255, 255, 255, 140))
    ring_pen.setWidthF(1.1)
    p.setPen(ring_pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QRectF(11 - 5.2, 9 - 5.2, 10.4, 10.4))
    p.end()

    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    buf = buf.reshape(CURSOR_SIZE, img.bytesPerLine())
    buf = buf[:, :CURSOR_SIZE * 4].reshape(CURSOR_SIZE, CURSOR_SIZE, 4)
    b, g, r, a = buf[..., 0], buf[..., 1], buf[..., 2], buf[..., 3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32), a.astype(np.float32) / 255.0


def _point_in_polygon(px, py, pts):
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _fallback_cursor():
    """Qt 不可用时的兜底：按同一轮廓光栅化，保证仍是光标而不是方块。"""
    gray = np.full((CURSOR_SIZE, CURSOR_SIZE), 232.0, dtype=np.float32)
    alpha = np.zeros((CURSOR_SIZE, CURSOR_SIZE), dtype=np.float32)
    inside = np.zeros((CURSOR_SIZE, CURSOR_SIZE), dtype=bool)
    scale = 24.0 / CURSOR_SIZE
    for y in range(CURSOR_SIZE):
        for x in range(CURSOR_SIZE):
            inside[y, x] = _point_in_polygon(
                (x + 0.5) * scale, (y + 0.5) * scale, _CURSOR_PATH)
    alpha[inside] = 1.0
    # 轮廓压暗，模拟 1.4px 深色描边
    border = inside & ~(
        np.roll(inside, 1, 0) & np.roll(inside, -1, 0)
        & np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
    )
    gray[border] = 11.0
    return gray, alpha


def load_or_render_cursor(cache_path):
    """优先读缓存；不存在则渲染并缓存。返回 (灰度 float32, alpha float32)。"""
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            with np.load(str(cache_path)) as z:
                gray, alpha = z["gray"], z["alpha"]
            if gray.shape == (CURSOR_SIZE, CURSOR_SIZE):
                return gray.astype(np.float32), alpha.astype(np.float32)
        except Exception:
            pass
    try:
        gray, alpha = _render_cursor_via_qt()
    except Exception:
        gray, alpha = _fallback_cursor()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(cache_path), gray=gray, alpha=alpha)
    except Exception:
        pass
    return gray, alpha


def _paste_cursor(crop, top, cursor_x, cursor_y, sprite):
    """把光标精灵按 alpha 混合贴进视野裁剪图，位置与 ChatUI 的 translate(-1px,-1px) 对齐。"""
    gray, alpha = sprite
    ox = int(round(cursor_x)) - 1
    oy = int(round(cursor_y - top)) - 1
    x0, y0 = max(0, ox), max(0, oy)
    x1 = min(SCREEN_W, ox + CURSOR_SIZE)
    y1 = min(VIEW_H, oy + CURSOR_SIZE)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - ox, y0 - oy
    sub_g = gray[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    sub_a = alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    region = crop[y0:y1, x0:x1].astype(np.float32)
    crop[y0:y1, x0:x1] = np.clip(
        region * (1.0 - sub_a) + sub_g * sub_a, 0, 255).astype(np.uint8)


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


def evaluate(cfg, syn, screen, ticks=TICKS_PER_RUN, cursor_overlay=True,
             cursor_sprite=None, compact=None):
    """
    跑一个网络 ticks 步，返回：
      mouse_positions: [(x,y), ...] 每 tick 的光标位置
      click_positions: [(x,y), ...] 发生点击的位置
      frames: [{cx,cy,vy,click}, ...] 供回放

    compact=(biases,roles,ipidx,a,b,w) 时直接数组建网，避免构造字典。
    """
    net = (drn_core.Network.from_arrays(*compact) if compact is not None
           else drn_core.Network(cfg, syn))
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
            if cursor_sprite is not None:
                _paste_cursor(crop, top, cursor_x, cursor_y, cursor_sprite)
            else:
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
