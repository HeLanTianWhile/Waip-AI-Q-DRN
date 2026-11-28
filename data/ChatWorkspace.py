# -*- coding: utf-8 -*-
"""ChatUI 工作区：像素/位置输入与鼠标/视野/点击输出的闭环。"""

import json
import math
import queue
import sys
import threading
import time
import traceback
from math import tanh
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QImage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
HTML_FILE = ROOT_DIR / "ChatUI" / "index.html"        # AI 端（网络像素输入唯一来源）
USER_HTML_FILE = ROOT_DIR / "ChatUI" / "user.html"    # 用户端（独立页面，不与 AI 端共用 UI/状态）

SCREEN_W, SCREEN_H = 390, 844
VIEW_W, VIEW_H = 390, 192
VIEW_X = SCREEN_W / 2.0
PIXEL_COUNT = VIEW_W * VIEW_H
INPUT_COUNT = PIXEL_COUNT + 4
OUTPUT_COUNT = 4
SAVE_INTERVAL = 15.0


def clamp(value, low, high):
    return max(low, min(high, value))


def normalized_position(x, y):
    """坐标除以屏幕尺寸，并四舍五入到三位小数。"""
    return round(x / SCREEN_W, 3), round(y / SCREEN_H, 3)


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)


class NeuralWorker(threading.Thread):
    """串行处理 GUI 采集到的画面；数据库只由该线程修改。"""

    def __init__(self, config_db, synapses_db, state_db, save_callback):
        super().__init__(daemon=True)
        self.config_db = config_db
        self.synapses_db = synapses_db
        self.state_db = state_db
        self.save_callback = save_callback
        self.signals = WorkerSignals()
        self.tasks = queue.Queue()
        self.stopping = threading.Event()
        self.input_ids = [None] * INPUT_COUNT
        self.output_ids = [None] * OUTPUT_COUNT
        self._collect_neurons()
        self.outgoing = self._build_reverse_index()
        self.ticks = 0
        self.rate = 0.0
        self._rate_count = 0
        self._rate_time = time.monotonic()
        self._save_time = time.monotonic()

    def _collect_neurons(self):
        input_indices = []
        output_indices = []
        for neuron_id, config in self.config_db.items():
            ip = config.get("IP")
            if not isinstance(ip, (list, tuple)) or len(ip) < 2:
                continue
            kind, index = ip[0], ip[1]
            if not isinstance(index, int):
                continue
            if kind == "Input":
                input_indices.append(index)
                if 0 <= index < INPUT_COUNT:
                    self.input_ids[index] = neuron_id
            elif kind == "Output":
                output_indices.append(index)
                if 0 <= index < OUTPUT_COUNT:
                    self.output_ids[index] = neuron_id

        if sorted(input_indices) != list(range(INPUT_COUNT)):
            raise ValueError(
                f"ChatUI 工作区需要 {INPUT_COUNT} 个输入神经元（390×192 像素 + 4 个位置），"
                f"当前为 {len(input_indices)} 个。请用 && create 新建最终配置的意识体。"
            )
        if sorted(output_indices) != list(range(OUTPUT_COUNT)):
            raise ValueError(
                f"ChatUI 工作区需要 4 个输出神经元（鼠标 X/Y、视野 Y、点击），"
                f"当前为 {len(output_indices)} 个。请用 && create 新建最终配置的意识体。"
            )

    def _build_reverse_index(self):
        """与 main.py 的突触语义一致：双向互传，单向只允许第二个元素传给第一个。"""
        outgoing = {}
        for synapse in self.synapses_db.values():
            targets = synapse.get("Connection Target")
            if not isinstance(targets, (list, tuple)) or len(targets) != 2:
                continue
            first, second = targets
            if first == second:
                continue
            weight = synapse.get("Weight", 0.0)
            bidirectional = synapse.get("Type", "bidirectional synapse") != "unidirectional synapse"
            if bidirectional:
                outgoing.setdefault(first, []).append((second, weight))
            outgoing.setdefault(second, []).append((first, weight))
        return outgoing

    def submit(self, pixels, cursor_x, cursor_y, view_y):
        self.tasks.put((pixels, cursor_x, cursor_y, view_y))

    def stop(self):
        self.stopping.set()
        self.tasks.put(None)

    def run(self):
        try:
            while not self.stopping.is_set():
                task = self.tasks.get()
                if task is None:
                    break
                result = self.process(*task)
                self.signals.result.emit(result)
        except Exception:
            traceback.print_exc()
            self.signals.error.emit(traceback.format_exc())
        finally:
            self._save()

    def process(self, pixels, cursor_x, cursor_y, view_y):
        if len(pixels) != PIXEL_COUNT:
            raise ValueError(f"视野像素应为 {PIXEL_COUNT} 个，实际为 {len(pixels)} 个")
        self._inject(pixels, cursor_x, cursor_y, view_y)
        self._sanitize_state()
        self._propagate()
        self._sanitize_state()
        outputs = self._read_outputs()

        self.ticks += 1
        self._rate_count += 1
        now = time.monotonic()
        elapsed = now - self._rate_time
        if elapsed >= 0.5:
            self.rate = self._rate_count / elapsed
            self._rate_count = 0
            self._rate_time = now
        if now - self._save_time >= SAVE_INTERVAL:
            self._save()
            self._save_time = now
        return {"outputs": outputs, "ticks": self.ticks, "rate": self.rate}

    def _inject(self, pixels, cursor_x, cursor_y, view_y):
        # 前 74,880 个输入：视野灰度，按从左到右、从上到下的顺序，0 表示不激活。
        for index, gray in enumerate(pixels):
            self.state_db[self.input_ids[index]]["y"] = (
                gray / 255.0 if gray else "NOT Activate"
            )

        # 后 4 个输入：视野 X/Y、鼠标 X/Y；均归一化并保留三位小数。
        positions = (
            *normalized_position(VIEW_X, view_y),
            *normalized_position(cursor_x, cursor_y),
        )
        for offset, value in enumerate(positions):
            self.state_db[self.input_ids[PIXEL_COUNT + offset]]["y"] = (
                float(value) if value else "NOT Activate"
            )

    def _propagate(self):
        active = []
        for neuron_id, state in self.state_db.items():
            y = state.get("y")
            if isinstance(y, (int, float)) and math.isfinite(y):
                active.append(neuron_id)

        dispatch = {}
        for source in active:
            value = self.state_db[source]["y"]
            for target, weight in self.outgoing.get(source, ()):
                dispatch[target] = dispatch.get(target, 0.0) + weight * value

        changes = {
            target: tanh(value - self.config_db[target]["bias"])
            for target, value in dispatch.items()
        }
        for source in active:
            if source not in changes:
                self.state_db[source]["y"] = "NOT Activate"
        for target, value in changes.items():
            self.state_db[target]["y"] = value

    def _sanitize_state(self):
        for state in self.state_db.values():
            y = state.get("y")
            if not isinstance(y, (int, float)):
                continue
            if math.isnan(y):
                state["y"] = "NOT Activate"
            elif math.isinf(y):
                state["y"] = 1.0 if y > 0 else -1.0

    def _read_outputs(self):
        outputs = []
        for neuron_id in self.output_ids:
            y = self.state_db[neuron_id].get("y")
            outputs.append(
                float(y) if isinstance(y, (int, float)) and math.isfinite(y) else 0.0
            )
        return outputs

    def _save(self):
        try:
            self.save_callback(self.config_db, self.state_db, self.synapses_db)
        except Exception:
            traceback.print_exc()


class ChatWorkspaceWindow(QMainWindow):
    def __init__(self, agent_name, worker):
        super().__init__()
        self.agent_name = agent_name
        self.worker = worker
        self.cursor_x = SCREEN_W / 2.0
        self.cursor_y = SCREEN_H / 2.0
        self.view_y = SCREEN_H / 2.0
        self.busy = False
        self.closing = False

        self.setWindowTitle(f"Waip AI Q-DRN · ChatUI — {agent_name}")
        self.running = True
        self.delay_ms = 30
        self.web = QWebEngineView(self)
        self.web.setFixedSize(SCREEN_W, SCREEN_H)
        self.web.setContextMenuPolicy(Qt.NoContextMenu)
        self.web.setFocusPolicy(Qt.NoFocus)
        # 左侧 self.web 是 AI 唯一会被截屏的页面；右侧仅供真人操作。
        self.user_web = QWebEngineView(self)
        self.user_web.setFixedSize(SCREEN_W, SCREEN_H)
        self.user_web.setContextMenuPolicy(Qt.NoContextMenu)

        bar = QWidget(self)
        bar.setFixedHeight(38)
        controls = QHBoxLayout(bar)
        controls.setContentsMargins(4, 3, 4, 3)
        controls.setSpacing(4)
        self.btn_run = QPushButton("⏸ 暂停")
        self.btn_reset_net = QPushButton("重置网络")
        self.btn_reset_ui = QPushButton("重置 ChatUI")
        self.btn_step = QPushButton("运行 1 tick")
        self.speed = QComboBox()
        for label, delay in (
            ("0.25×", 120),
            ("0.5×", 60),
            ("1×", 30),
            ("2×", 15),
            ("4×", 0),
        ):
            self.speed.addItem(label, delay)
        self.speed.setCurrentIndex(2)
        for widget in (
            self.btn_run,
            self.btn_reset_net,
            self.btn_reset_ui,
            self.btn_step,
            self.speed,
        ):
            widget.setFocusPolicy(Qt.NoFocus)
            controls.addWidget(widget)

        panes = QWidget(self)
        panes_layout = QHBoxLayout(panes)
        panes_layout.setContentsMargins(0, 0, 0, 0)
        panes_layout.setSpacing(0)
        panes_layout.addWidget(self.web)
        panes_layout.addWidget(self.user_web)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(panes)
        layout.addWidget(bar)
        self.setCentralWidget(container)
        self.setFixedSize(SCREEN_W * 2, SCREEN_H + 38)

        self.btn_run.clicked.connect(self._toggle_run)
        self.btn_reset_net.clicked.connect(self._reset_network)
        self.btn_reset_ui.clicked.connect(self._reset_chat_ui)
        self.btn_step.clicked.connect(self._step_tick)
        self.speed.currentIndexChanged.connect(self._change_speed)

        self.web.loadFinished.connect(self._loaded)
        self.user_web.loadFinished.connect(self._user_loaded)
        self.web.load(QUrl.fromLocalFile(str(HTML_FILE)))
        # 用户端加载独立的 user.html：正常光标、不拦截真人事件、气泡按“我/对方”对齐，
        # 与 AI 端 index.html 完全隔离，仅通过 qdrnTakeOutbox/qdrnAppendMessage 交换消息。
        self.user_web.load(QUrl.fromLocalFile(str(USER_HTML_FILE)))

        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._sync_messages)
        self._sync_timer.start(80)

    def _toggle_run(self):
        self.running = not self.running
        self.btn_run.setText("⏸ 暂停" if self.running else "▶ 运行")
        if self.running:
            self.capture()

    def _step_tick(self):
        if not self.running:
            self.capture(force=True)

    def _change_speed(self):
        self.delay_ms = int(self.speed.currentData())

    def _reset_network(self):
        if self.busy:
            QTimer.singleShot(50, self._reset_network)
            return
        for state in self.worker.state_db.values():
            state["y"] = "NOT Activate"
        self.worker.ticks = 0
        self.setWindowTitle(
            f"Waip AI Q-DRN · ChatUI — {self.agent_name} | 已重置网络 | tick 0"
        )

    def _reset_chat_ui(self):
        self.cursor_x = SCREEN_W / 2.0
        self.cursor_y = SCREEN_H / 2.0
        self.view_y = SCREEN_H / 2.0
        self.web.reload()
        self.user_web.reload()

    def _user_loaded(self, ok):
        if not ok and not self.closing:
            self.fail("无法加载用户端 ChatUI")

    def _sync_messages(self):
        """同步两侧的本地发送记录；AI 的像素输入仍只从左侧 self.web 截取。"""
        if self.closing:
            return
        for source, target in (
            (self.web, self.user_web),
            (self.user_web, self.web),
        ):
            source.page().runJavaScript(
                "qdrnTakeOutbox()",
                lambda result, target=target: self._forward_messages(result, target),
            )

    def _forward_messages(self, result, target):
        if self.closing or not isinstance(result, str):
            return
        try:
            messages = json.loads(result)
        except (TypeError, ValueError):
            return
        for message in messages:
            role = "user" if message.get("role") == "user" else "ai"
            text = str(message.get("text", ""))
            script = (
                f"qdrnAppendMessage({json.dumps(role)}, "
                f"{json.dumps(text, ensure_ascii=False)});"
            )
            target.page().runJavaScript(script)

    def _loaded(self, ok):
        if not ok:
            self.fail(f"无法加载 {HTML_FILE}")
            return
        # 采样窗网格是给人看的叠层，网络不应看见它，否则会干扰视觉判断。
        self.web.page().runJavaScript(
            "document.getElementById('aperture').style.display='none';"
        )
        QTimer.singleShot(150, self.capture)

    def capture(self, force=False):
        if self.busy or self.closing:
            return
        if not self.running and not force:
            return
        image = self.web.grab().toImage()
        if image.isNull():
            QTimer.singleShot(30, self.capture)
            return
        image = image.scaled(
            SCREEN_W, SCREEN_H, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        top = int(round(self.view_y - VIEW_H / 2.0))
        view = image.copy(0, top, VIEW_W, VIEW_H).convertToFormat(
            QImage.Format_Grayscale8
        )
        bits = memoryview(view.constBits())
        stride = view.bytesPerLine()
        pixels = b"".join(
            bytes(bits[row * stride:row * stride + VIEW_W])
            for row in range(VIEW_H)
        )
        self.busy = True
        self.worker.submit(pixels, self.cursor_x, self.cursor_y, self.view_y)

    def apply_result(self, result):
        if self.closing:
            return
        outputs = result["outputs"]
        # 输出 0/1：鼠标 X/Y；输出 2：视野 Y；输出 3：大于 0 时点击一次。
        self.cursor_x = clamp(
            self.cursor_x + SCREEN_W * outputs[0], 0.0, float(SCREEN_W)
        )
        self.cursor_y = clamp(
            self.cursor_y + SCREEN_H * outputs[1], 0.0, float(SCREEN_H)
        )
        self.view_y = clamp(
            self.view_y + SCREEN_H * outputs[2],
            VIEW_H / 2.0,
            SCREEN_H - VIEW_H / 2.0,
        )
        click = outputs[3] > 0.0
        script = (
            f"qdrnSetState({self.cursor_x:.6f}, {self.cursor_y:.6f}, {self.view_y:.6f});"
        )
        if click:
            # 所有点击都派发到光标下的真实控件（软键盘 / 发送按钮），
            # 不再有任何 Python 端的文本注入捷径——发消息必须由网络逐键点击完成。
            script += f"qdrnClick({self.cursor_x:.6f}, {self.cursor_y:.6f});"
        self.setWindowTitle(
            f"Waip AI Q-DRN · ChatUI — {self.agent_name} | "
            f"{result['rate']:.1f} tick/s | tick {result['ticks']}"
        )
        self.busy = False
        if self.running and not self.closing:
            self.web.page().runJavaScript(
                script,
                lambda _result: QTimer.singleShot(self.delay_ms, self.capture),
            )
        else:
            self.web.page().runJavaScript(script)

    def fail(self, message):
        print(f"[错误] ChatUI 工作区：{message}")
        self.close()

    def closeEvent(self, event):
        if not self.closing:
            self.closing = True
            self._sync_timer.stop()
            self.worker.stop()
            self.worker.join(30)
            if self.worker.is_alive():
                print("[警告] 神经网络线程 30 秒内未退出，最新状态可能尚未保存。")
        event.accept()


def run_workspace(agent_name, config_db, synapses_db, state_db, _main_func, save_callback):
    """打开固定的 390×844 ChatUI 窗口；main.py 本身不需要修改。"""
    sys.setswitchinterval(0.001)
    app = QApplication.instance() or QApplication(sys.argv)
    worker = NeuralWorker(config_db, synapses_db, state_db, save_callback)
    window = ChatWorkspaceWindow(agent_name, worker)
    worker.signals.result.connect(window.apply_result)
    worker.signals.error.connect(window.fail)
    worker.start()
    window.show()
    code = app.exec()
    worker.stop()
    worker.join(30)
    return code
