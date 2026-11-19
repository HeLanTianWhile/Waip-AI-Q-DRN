# -*- coding: utf-8 -*-
"""Waip AI 训练场。

- 宠物在深色场地内，瞳孔朝向就是 heading，不跟随鼠标，初始 0°（+X）；
- 每轮前向传播结束后立即应用三个输出神经元（X、Y、转向）；
- 随后用 heading ± 50°（共 100°）的光线追踪生成下一轮输入；
- 场地内随机放置圆形物体；光线命中物体时按距离换算成 0~1 输入（越近越接近 1），没命中为 0；
"""

import contextlib
import io
import math
import random
import sys
import threading
import time
import traceback

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

EYE_R = 60.0                       # 眼睛半径（逻辑像素）
FOV_DEG = 100.0                    # 视野总角度
UNIT_PX = 1.0                      # 输出 1 个单位对应的位移（逻辑像素）
SAVE_INTERVAL = 15.0               # 数据库周期性回写间隔（秒）
W_COUNT = 16                       # 环绕的 w 数量

MAX_PUPIL = 0.5 * EYE_R            # 瞳孔最大偏移
W_ORBIT = 1.47 * EYE_R             # w 环绕半径
W_FONT = 0.21 * EYE_R              # w 字号
S_NEAR, S_FAR = 2.2, 0.6           # 近大远小
NEAR_D = 0.97 * EYE_R
FAR_D = 1.55 * EYE_R
BORDER = 0.063 * EYE_R             # 眼睛边框

OBJECT_COUNT = 8                   # 场地内随机物体数量
OBJECT_RADIUS_MIN = 12.0
OBJECT_RADIUS_MAX = 34.0

BACKGROUND = QColor("#0b0f16")
EYE_FILL = QColor("#ffffff")
EYE_EDGE = QColor("#000000")
W_COLOR = QColor("#dbe4f0")
OBJECT_FILL = QColor("#c98a5b")
OBJECT_EDGE = QColor("#8a5a34")
HUD_COLOR = QColor("#8b98ab")


def _saturate(value):
    """输出按规格视为 -1..1；超出部分饱和，避免一步飞出场地或转出无数圈。"""
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def generate_objects(arena_w, arena_h, count=OBJECT_COUNT):
    """在场地内随机放置圆形物体；正前方保证有一个，用于冷激活。"""
    rng = random.Random()
    objects = []
    cx = arena_w / 2.0
    cy = arena_h / 2.0

    # 初始 heading 为 0°（+X），在正前方放一个物体，保证第一轮必有非零输入
    if count > 0:
        radius = rng.uniform(OBJECT_RADIUS_MIN, OBJECT_RADIUS_MAX)
        near = EYE_R + radius + 40.0
        far = arena_w - radius - 8.0 - cx
        if far > near:
            # 放在最近合法位置，输入值接近 1，保证冷激活足够强
            objects.append((cx + near, cy, radius))

    for _ in range(max(0, count - len(objects))):
        for _ in range(200):
            radius = rng.uniform(OBJECT_RADIUS_MIN, OBJECT_RADIUS_MAX)
            x = rng.uniform(radius + 8.0, arena_w - radius - 8.0)
            y = rng.uniform(radius + 8.0, arena_h - radius - 8.0)
            if math.hypot(x - cx, y - cy) < EYE_R + radius + 40.0:
                continue
            if any(
                math.hypot(x - ox, y - oy) < radius + other + 10.0
                for ox, oy, other in objects
            ):
                continue
            objects.append((x, y, radius))
            break
    return tuple(objects)


def collect_input_indices(config_db):
    """返回排序去重后的输入神经元 IP 索引。"""
    indices = set()
    for config in config_db.values():
        ip = config.get("IP")
        if (
            isinstance(ip, (list, tuple))
            and len(ip) >= 2
            and ip[0] == "Input"
            and isinstance(ip[1], int)
        ):
            indices.add(ip[1])
    return sorted(indices)


def read_outputs(config_db, state_db):
    """读取 Output 0/1/2 的激活值，缺失或非数字按 0。"""
    outputs = [0.0, 0.0, 0.0]
    for neuron_id, config in config_db.items():
        ip = config.get("IP")
        if not isinstance(ip, (list, tuple)) or len(ip) < 2 or ip[0] != "Output":
            continue
        index = ip[1]
        if not isinstance(index, int) or not (0 <= index <= 2):
            continue
        y = state_db.get(neuron_id, {}).get("y", "NOT Activate")
        if isinstance(y, (int, float)) and math.isfinite(y):
            outputs[index] = float(y)
    return outputs


class PetModel:
    """宠物的纯逻辑状态，只有神经网络线程会修改它。"""

    def __init__(self, arena_w, arena_h, input_indices, objects=()):
        self.arena_w = max(2 * EYE_R, float(arena_w))
        self.arena_h = max(2 * EYE_R, float(arena_h))
        self.input_indices = list(input_indices)
        self.objects = tuple(objects)
        self.x = self.arena_w / 2.0
        self.y = self.arena_h / 2.0
        self.heading_total = 0.0
        self.outputs = [0.0, 0.0, 0.0]

    @property
    def heading(self):
        return self.heading_total % 360.0

    def set_arena(self, arena_w, arena_h):
        self.arena_w = max(2 * EYE_R, float(arena_w))
        self.arena_h = max(2 * EYE_R, float(arena_h))
        self.clamp()

    def clamp(self):
        self.x = min(max(self.x, EYE_R), self.arena_w - EYE_R)
        self.y = min(max(self.y, EYE_R), self.arena_h - EYE_R)

    def apply(self, outputs):
        """X、Y、转向三个输出立刻作用到位置和朝向上。"""
        self.outputs = list(outputs)
        self.x += _saturate(outputs[0]) * UNIT_PX
        self.y += _saturate(outputs[1]) * UNIT_PX
        self.heading_total += _saturate(outputs[2]) * 360.0
        self.clamp()

    def sense(self):
        """100° 视野内逐条光线求交：命中物体越近，输入值越接近 1；没命中为 0。"""
        count = len(self.input_indices)
        if count == 0:
            return {}
        values = {}
        half = FOV_DEG / 2.0
        if count == 1:
            angles = [self.heading]
        else:
            angles = [self.heading - half + FOV_DEG * i / (count - 1) for i in range(count)]
        for index, angle in zip(self.input_indices, angles):
            values[index] = self._cast_ray(math.radians(angle))
        return values

    def _border_distance(self, angle):
        """光线到窗口矩形的距离。"""
        dx, dy = math.cos(angle), math.sin(angle)
        if dx > 0:
            tx = (self.arena_w - self.x) / dx
        elif dx < 0:
            tx = (0.0 - self.x) / dx
        else:
            tx = math.inf
        if dy > 0:
            ty = (self.arena_h - self.y) / dy
        elif dy < 0:
            ty = (0.0 - self.y) / dy
        else:
            ty = math.inf
        distance = min(tx, ty)
        if not math.isfinite(distance) or distance < 0:
            return 0.0
        return distance

    def _cast_ray(self, angle):
        """最近命中距离换算成 0~1；没命中返回 0。"""
        border = self._border_distance(angle)
        if border <= 0.0:
            return 0.0
        dx, dy = math.cos(angle), math.sin(angle)
        nearest = border
        hit = False
        for ox, oy, radius in self.objects:
            fx = self.x - ox
            fy = self.y - oy
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - radius * radius
            discriminant = b * b - 4.0 * c
            if discriminant < 0.0:
                continue
            root = math.sqrt(discriminant)
            for t in ((-b - root) / 2.0, (-b + root) / 2.0):
                if 0.0 < t < nearest:
                    nearest = t
                    hit = True
                    break
        if not hit:
            return 0.0
        return max(0.0, min(1.0, 1.0 - nearest / border))


class NeuralWorker(threading.Thread):
    """以“传播完立刻应用输出、立刻重新感知”的自定节奏运行。"""

    def __init__(self, model, config_db, synapses_db, state_db, main_func,
                 save_callback, stop_event, lock, snapshot):
        super().__init__(daemon=True)
        self.model = model
        self.config_db = config_db
        self.synapses_db = synapses_db
        self.state_db = state_db
        self.main_func = main_func
        self.save_callback = save_callback
        self.stop_event = stop_event
        self.lock = lock
        self.snapshot = snapshot
        self.inputs = model.sense()
        self.input_neuron_ids = {}
        for neuron_id, config in config_db.items():
            ip = config.get("IP")
            if (
                isinstance(ip, (list, tuple))
                and len(ip) >= 2
                and ip[0] == "Input"
                and isinstance(ip[1], int)
            ):
                self.input_neuron_ids[ip[1]] = neuron_id
        self._sanitize_state()
        self.ticks = 0
        self.rate = 0.0
        self._rate_count = 0
        self._rate_time = time.monotonic()
        self._save_time = time.monotonic()

    def run(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    arena_w = self.snapshot["arena_w"]
                    arena_h = self.snapshot["arena_h"]
                self.model.set_arena(arena_w, arena_h)

                self._inject(self.inputs)
                self.config_db, self.state_db, self.synapses_db = self._propagate()
                self._sanitize_state()
                outputs = read_outputs(self.config_db, self.state_db)
                self.model.apply(outputs)
                self.inputs = self.model.sense()
                hits = sum(1 for value in self.inputs.values() if value > 0.0)

                self.ticks += 1
                self._rate_count += 1
                now = time.monotonic()
                if now - self._rate_time >= 0.5:
                    self.rate = self._rate_count / (now - self._rate_time)
                    self._rate_count = 0
                    self._rate_time = now
                if now - self._save_time >= SAVE_INTERVAL:
                    self._save()
                    self._save_time = now

                with self.lock:
                    self.snapshot.update(
                        x=self.model.x,
                        y=self.model.y,
                        heading=self.model.heading,
                        outputs=tuple(outputs),
                        hits=hits,
                        ticks=self.ticks,
                        rate=self.rate,
                    )

                if self.ticks % 8 == 0:
                    time.sleep(0)
        except Exception:
            traceback.print_exc()
        finally:
            self._save()

    def _inject(self, values):
        """把这一轮感知到的 0~1 距离值写进对应输入神经元；0 表示不激活。"""
        for index, neuron_id in self.input_neuron_ids.items():
            value = values.get(index, 0.0)
            if value > 0.0:
                self.state_db[neuron_id]["y"] = float(value)
            else:
                self.state_db[neuron_id]["y"] = "NOT Activate"

    def _sanitize_state(self):
        """闭环传播会累积出 inf/NaN，映射成 ±1 或关闭，避免输出被 read_outputs 当成 0。"""
        for value in self.state_db.values():
            y = value.get("y")
            if isinstance(y, (int, float)):
                if math.isnan(y):
                    value["y"] = "NOT Activate"
                elif math.isinf(y):
                    value["y"] = 1.0 if y > 0 else -1.0

    def _propagate(self):
        # main/main.py 只接受单个输入索引，这里先直接把感知值写进输入神经元，
        # 再以 <blank> 调用，避免 +1 的单位输入。main() 里的 print 会刷屏，临时吞掉。
        with contextlib.redirect_stdout(io.StringIO()):
            return self.main_func(
                self.config_db, self.synapses_db, self.state_db, "<blank>"
            )

    def _save(self):
        try:
            self.save_callback(self.config_db, self.state_db, self.synapses_db)
        except Exception:
            traceback.print_exc()


class TrainingWindow(QWidget):
    def __init__(self, agent_name, snapshot, lock, stop_event, worker):
        super().__init__()
        self.agent_name = agent_name
        self.snapshot = snapshot
        self.lock = lock
        self.stop_event = stop_event
        self.worker = worker
        self.view_x = snapshot["x"]
        self.view_y = snapshot["y"]
        self.view_heading = snapshot["heading"]
        self.setWindowTitle(f"Waip AI 训练场 — {agent_name}")
        self.resize(900, 600)
        self.setMinimumSize(480, 360)
        self.setFocusPolicy(Qt.StrongFocus)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def resizeEvent(self, event):
        with self.lock:
            self.snapshot["arena_w"] = float(self.width())
            self.snapshot["arena_h"] = float(self.height())
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.stop_event.set()
        self.worker.join(5)
        if self.worker.is_alive():
            print("[警告] 神经网络线程 5 秒内未退出，数据可能未保存到最新状态。")
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        painter.fillRect(self.rect(), BACKGROUND)

        with self.lock:
            state = dict(self.snapshot)

        # 场地里的随机物体
        painter.setPen(QPen(OBJECT_EDGE, 2.0))
        painter.setBrush(OBJECT_FILL)
        for ox, oy, radius in state.get("objects", ()):
            painter.drawEllipse(QPointF(ox, oy), radius, radius)

        # 显示层平滑：神经网络 tick 比 60fps 慢，插值让位移/转向可见
        self.view_x += (state["x"] - self.view_x) * 0.25
        self.view_y += (state["y"] - self.view_y) * 0.25
        delta = ((state["heading"] - self.view_heading + 180.0) % 360.0) - 180.0
        self.view_heading = (self.view_heading + delta * 0.25) % 360.0

        cx, cy = self.view_x, self.view_y
        # 瞳孔朝向 = heading，不跟随鼠标
        pupil_x = math.cos(math.radians(self.view_heading)) * MAX_PUPIL
        pupil_y = math.sin(math.radians(self.view_heading)) * MAX_PUPIL

        # 16 个 w：绕宠物中心均布，整体跟随 heading 转动，按到瞳孔的距离缩放
        base = math.radians(self.view_heading)
        for i in range(W_COUNT):
            angle = base + 2.0 * math.pi * i / W_COUNT
            wx = cx + math.cos(angle) * W_ORBIT
            wy = cy + math.sin(angle) * W_ORBIT
            d = math.hypot(
                math.cos(angle) * W_ORBIT - pupil_x,
                math.sin(angle) * W_ORBIT - pupil_y,
            )
            n = min(max((d - NEAR_D) / (FAR_D - NEAR_D), 0.0), 1.0)
            scale = S_NEAR + (S_FAR - S_NEAR) * n
            font = QFont("Segoe UI")
            font.setPixelSize(max(6, int(round(W_FONT * scale))))
            painter.setFont(font)
            painter.setPen(W_COLOR)
            size = W_FONT * scale * 2.0
            painter.drawText(
                QRectF(wx - size, wy - size, size * 2.0, size * 2.0),
                Qt.AlignCenter,
                "w",
            )

        # 眼睛
        painter.setPen(QPen(EYE_EDGE, BORDER))
        painter.setBrush(EYE_FILL)
        painter.drawEllipse(QPointF(cx, cy), EYE_R, EYE_R)

        # 两条竖线瞳孔
        px = cx + pupil_x
        py = cy + pupil_y
        line_w = 0.095 * EYE_R
        line_h = 0.59 * EYE_R
        offset = (0.42 * EYE_R + line_w) / 2.0
        painter.setPen(QPen(EYE_EDGE, line_w, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(px - offset, py - line_h / 2.0),
                         QPointF(px - offset, py + line_h / 2.0))
        painter.drawLine(QPointF(px + offset, py - line_h / 2.0),
                         QPointF(px + offset, py + line_h / 2.0))

        # HUD
        outputs = state["outputs"]
        applied = tuple(_saturate(value) for value in outputs)
        lines = [
            f"Agent: {self.agent_name}",
            f"tick/s: {state['rate']:.0f}    ticks: {state['ticks']}    "
            f"hits: {state.get('hits', 0)} rays",
            f"x={state['x']:.1f}  y={state['y']:.1f}  heading={state['heading']:.1f}°",
            " ".join(f"out{i}={outputs[i]:+.4g}" for i in range(3)),
            "apply dx=%+.2f dy=%+.2f dθ=%+.1f°"
            % (applied[0], applied[1], applied[2] * 360.0),
        ]
        hud_font = QFont("Microsoft YaHei")
        hud_font.setPixelSize(13)
        painter.setFont(hud_font)
        painter.setPen(HUD_COLOR)
        for i, line in enumerate(lines):
            painter.drawText(14, 26 + i * 19, line)


def run_trainer(agent_name, config_db, synapses_db, state_db, main_func, save_callback):
    """打开训练场窗口，直到窗口关闭才返回。"""
    sys.setswitchinterval(0.001)
    app = QApplication.instance() or QApplication(sys.argv)

    input_indices = collect_input_indices(config_db)
    objects = generate_objects(900.0, 600.0)
    model = PetModel(900.0, 600.0, input_indices, objects)

    lock = threading.Lock()
    snapshot = {
        "x": model.x,
        "y": model.y,
        "heading": model.heading,
        "outputs": (0.0, 0.0, 0.0),
        "hits": 0,
        "objects": objects,
        "ticks": 0,
        "rate": 0.0,
        "arena_w": 900.0,
        "arena_h": 600.0,
    }
    stop_event = threading.Event()

    worker = NeuralWorker(
        model, config_db, synapses_db, state_db,
        main_func, save_callback, stop_event, lock, snapshot,
    )
    window = TrainingWindow(agent_name, snapshot, lock, stop_event, worker)
    with lock:
        snapshot["arena_w"] = float(window.width())
        snapshot["arena_h"] = float(window.height())

    worker.start()
    window.show()
    code = app.exec()
    stop_event.set()
    worker.join(5)
    return code
