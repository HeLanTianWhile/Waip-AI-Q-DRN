# -*- coding: utf-8 -*-
"""
Waip AI Q-DRN 控制台（启动弹窗）。

终端启动后不再直接运行，而是弹出本控制台，提供两个入口：
  · 训练 —— 启动 GA 训练（课程由 data/class/class-N.json 定义），面板内可实时
            查看训练日志、管理当前代网络、并“观看”任一回放的点击过程。
  · 运行模型 —— 选择一个已训练的意识体，以旧方式打开 ChatUI 工作区观看运行。
"""

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QFont, QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QStackedWidget,
    QSplitter, QAbstractItemView, QMessageBox, QComboBox, QInputDialog,
    QDoubleSpinBox, QSpinBox, QFormLayout, QDialog, QRadioButton,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
import numpy as np

import Simulator
import Trainer as TrainerMod
from Trainer import Trainer, TRAIN_DIR

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
HTML_FILE = ROOT_DIR / "ChatUI" / "index.html"

DARK = "#0e0e12"
PANEL = "#16161c"
ACCENT = "#7c5cff"
TEXT = "#e8e8f0"
MUTED = "#8a8a99"


def _style():
    return f"""
    QMainWindow, QWidget {{ background:{DARK}; color:{TEXT}; font-family:'Microsoft YaHei','Segoe UI',sans-serif; }}
    QPushButton {{ background:{PANEL}; color:{TEXT}; border:1px solid #2a2a36; border-radius:8px; padding:10px 18px; font-size:14px; }}
    QPushButton:hover {{ border-color:{ACCENT}; }}
    QPushButton:disabled {{ color:#555; border-color:#222; }}
    QListWidget, QPlainTextEdit, QTableWidget {{ background:#0a0a0e; color:{TEXT}; border:1px solid #2a2a36; border-radius:8px; }}
    QHeaderView::section {{ background:{PANEL}; color:{MUTED}; border:none; padding:6px; }}
    QLabel {{ color:{TEXT}; }}
    QComboBox {{ background:{PANEL}; color:{TEXT}; border:1px solid #2a2a36; padding:4px; }}
    """


# ===================== 回放窗口 =====================

class ReplayWindow(QMainWindow):
    """播放某个网络的点击回放（帧数由课程时间窗合计决定）。"""

    def __init__(self, traj, net_name, score, hit_w, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"观看回放 — {net_name}")
        self.resize(430, 940)
        self.frames = traj.get("frames", [])
        self._idx = 0
        self._click_count = sum(1 for f in self.frames if f.get("click"))
        self._hit = hit_w
        self._net_name = net_name
        self._score = score

        central = QWidget(); self.setCentralWidget(central)
        v = QVBoxLayout(central); v.setContentsMargins(12, 12, 12, 12)

        self.info = QLabel()
        self.info.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        v.addWidget(self.info)

        self.view = QWebEngineView()
        self.view.resize(390, 844)
        self.view.setUrl(QUrl.fromLocalFile(str(HTML_FILE)))
        v.addWidget(self.view, alignment=Qt.AlignCenter)

        ctrl = QHBoxLayout()
        self.speed = QComboBox()
        for s in ["0.5x", "1x", "2x", "4x"]:
            self.speed.addItem(s)
        self.speed.setCurrentText("1x")
        self.speed.currentTextChanged.connect(self._apply_speed)
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.clicked.connect(self.toggle_play)
        self.restart_btn = QPushButton("⟲ 重播")
        self.restart_btn.clicked.connect(self.restart)
        ctrl.addWidget(self.play_btn); ctrl.addWidget(self.restart_btn)
        ctrl.addStretch(); ctrl.addWidget(QLabel("速度")); ctrl.addWidget(self.speed)
        v.addLayout(ctrl)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._apply_speed()
        self.view.loadFinished.connect(lambda _ok: QTimer.singleShot(400, self.start))
        self._update_info()

    def _apply_speed(self):
        factor = float(self.speed.currentText().replace("x", ""))
        self._timer.setInterval(int(40 / factor))

    def toggle_play(self):
        if self._timer.isActive():
            self._timer.stop(); self.play_btn.setText("▶ 播放")
        else:
            self.start()

    def start(self):
        self._timer.start()
        self.play_btn.setText("⏸ 暂停")

    def restart(self):
        self._idx = 0
        self._update_info()
        self.start()

    def _step(self):
        if self._idx >= len(self.frames):
            self._timer.stop(); self.play_btn.setText("▶ 播放")
            return
        f = self.frames[self._idx]
        cx, cy, vy = f["cx"], f["cy"], f["vy"]
        self.view.page().runJavaScript(f"qdrnSetState({cx},{cy},{vy});")
        if f.get("click"):
            self.view.page().runJavaScript(f"qdrnClick({cx},{cy});")
        self._idx += 1
        self._update_info()

    def _update_info(self):
        score_txt = f"{self._score:+.4f}" if isinstance(self._score, (int, float)) else "—"
        self.info.setText(
            f"{self._net_name}   "
            f"tick {min(self._idx, len(self.frames))}/{len(self.frames)}   "
            f"得分 {score_txt}   点击 {self._click_count} 次   "
            f"命中 W 键: {'是' if self._hit else '否'}"
        )

    def closeEvent(self, e):
        self._timer.stop()
        super().closeEvent(e)


# ===================== 课程编辑器 =====================

class LessonEditorWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("课程编辑器")
        self.resize(980, 780)
        self.class_dir = DATA_DIR / "class"
        self.class_dir.mkdir(parents=True, exist_ok=True)
        self._score_mod = TrainerMod._load_main_module("drn_score_editor", "score.py")
        self.steps = []
        self.current = -1
        self._updating = False
        self._scale = 0.62

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        pick = QHBoxLayout()
        self.lesson_box = QComboBox()
        self.lesson_box.currentTextChanged.connect(self.load_lesson)
        pick.addWidget(self.lesson_box, 1)
        b_new = QPushButton("新建")
        b_new.clicked.connect(self.new_lesson)
        b_del = QPushButton("删除")
        b_del.clicked.connect(self.delete_lesson)
        pick.addWidget(b_new)
        pick.addWidget(b_del)
        left.addLayout(pick)

        self.step_list = QListWidget()
        self.step_list.currentRowChanged.connect(self.on_select)
        left.addWidget(self.step_list, 1)

        row1 = QHBoxLayout()
        b_add = QPushButton("＋ 添加步骤")
        b_add.clicked.connect(self.add_step)
        b_rm = QPushButton("－ 删除步骤")
        b_rm.clicked.connect(self.del_step)
        row1.addWidget(b_add)
        row1.addWidget(b_rm)
        left.addLayout(row1)
        row2 = QHBoxLayout()
        b_up = QPushButton("↑ 上移")
        b_up.clicked.connect(lambda: self.move_step(-1))
        b_down = QPushButton("↓ 下移")
        b_down.clicked.connect(lambda: self.move_step(1))
        row2.addWidget(b_up)
        row2.addWidget(b_down)
        left.addLayout(row2)
        root.addLayout(left, 0)

        right = QVBoxLayout()
        form = QFormLayout()
        self.type_box = QComboBox()
        self.type_box.addItems(["move_view", "mouse_move", "click"])
        self.type_box.currentTextChanged.connect(self.on_type)
        form.addRow("类型", self.type_box)
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(0, 390)
        self.x_spin.setDecimals(2)
        form.addRow("x（0~390）", self.x_spin)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(0, 844)
        self.y_spin.setDecimals(2)
        form.addRow("y（0~844，视野 96~748）", self.y_spin)
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1, 400)
        self.tol_spin.setDecimals(1)
        form.addRow("容差 tol（像素）", self.tol_spin)
        self.tick_spin = QSpinBox()
        self.tick_spin.setRange(1, 500)
        form.addRow("时间窗 ticks", self.tick_spin)
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0.1, 20)
        self.weight_spin.setDecimals(1)
        self.weight_spin.setSingleStep(0.5)
        form.addRow("权重 weight", self.weight_spin)
        for w in (self.x_spin, self.y_spin, self.tol_spin, self.tick_spin, self.weight_spin):
            w.valueChanged.connect(self.form_changed)
        right.addLayout(form)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        right.addWidget(self.preview, 1)
        self.ticks_label = QLabel()
        self.ticks_label.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        right.addWidget(self.ticks_label)
        b_save = QPushButton("💾 保存课程")
        b_save.setStyleSheet(f"background:#1c1633; border:2px solid {ACCENT}; font-size:16px;")
        b_save.clicked.connect(self.save)
        right.addWidget(b_save)
        root.addLayout(right, 1)

        self._base_pixmap = self._make_base()
        self.refresh_lessons()

    def _make_base(self):
        screen_path = TRAIN_DIR / "screen.npy"
        if screen_path.exists():
            arr = np.load(screen_path).astype(np.uint8)
        else:
            arr = np.full((844, 390), 18, dtype=np.uint8)
        img = QImage(arr.data, arr.shape[1], arr.shape[0],
                     arr.strides[0], QImage.Format_Grayscale8).copy()
        return QPixmap.fromImage(img).scaled(
            int(390 * self._scale), int(844 * self._scale),
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    def refresh_lessons(self, select=None):
        self.lesson_box.blockSignals(True)
        self.lesson_box.clear()
        names = sorted(p.name for p in self.class_dir.glob("class-*.json"))
        self.lesson_box.addItems(names)
        target = select or ("class-1.json" if "class-1.json" in names else (names[0] if names else None))
        if target:
            self.lesson_box.setCurrentText(target)
        self.lesson_box.blockSignals(False)
        if target:
            self.load_lesson(target)
        else:
            self.steps = []
            self.refresh_steps()

    def load_lesson(self, name):
        if not name:
            return
        path = self.class_dir / name
        try:
            self.steps = self._score_mod.load_lesson(path)
        except Exception as e:
            QMessageBox.critical(self, "课程读取失败", f"{name}\n{e}")
            self.steps = []
        self.current = 0 if self.steps else -1
        self.refresh_steps()

    def _step_text(self, i, st):
        pos = f"y={st['y']:g}" if st["type"] == "move_view" else f"({st['x']:g}, {st['y']:g})"
        return (f"{i+1}. {st['type']}  {pos}  tol={st['tol']:g}  "
                f"ticks={st['ticks']}  w={st['weight']:g}")

    def refresh_steps(self):
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for i, st in enumerate(self.steps):
            self.step_list.addItem(self._step_text(i, st))
        self.step_list.blockSignals(False)
        if 0 <= self.current < len(self.steps):
            self.step_list.setCurrentRow(self.current)
        else:
            self.on_select(-1)
        self.update_ticks_label()
        self.update_preview()

    def on_select(self, row):
        self.current = row
        self._updating = True
        if 0 <= row < len(self.steps):
            st = self.steps[row]
            self.type_box.setCurrentText(st["type"])
            self.x_spin.setValue(st.get("x", 0.0))
            self.y_spin.setValue(st["y"])
            self.tol_spin.setValue(st["tol"])
            self.tick_spin.setValue(st["ticks"])
            self.weight_spin.setValue(st["weight"])
        self._updating = False
        self._update_x_enabled()
        self.update_preview()

    def _update_x_enabled(self):
        self.x_spin.setEnabled(self.type_box.currentText() != "move_view")

    def on_type(self, kind):
        self._update_x_enabled()
        if self._updating or not (0 <= self.current < len(self.steps)):
            return
        st = self.steps[self.current]
        st["type"] = kind
        if kind != "move_view" and "x" not in st:
            st["x"] = 195.0
        self._sync_row()

    def form_changed(self):
        if self._updating or not (0 <= self.current < len(self.steps)):
            return
        st = self.steps[self.current]
        st["type"] = self.type_box.currentText()
        st["y"] = float(self.y_spin.value())
        if st["type"] != "move_view":
            st["x"] = float(self.x_spin.value())
        else:
            st.pop("x", None)
        st["tol"] = float(self.tol_spin.value())
        st["ticks"] = int(self.tick_spin.value())
        st["weight"] = float(self.weight_spin.value())
        self._sync_row()

    def _sync_row(self):
        i = self.current
        self.step_list.item(i).setText(self._step_text(i, self.steps[i]))
        self.update_ticks_label()
        self.update_preview()

    def add_step(self):
        self.steps.append({"type": "mouse_move", "x": 195.0, "y": 422.0,
                           "tol": 20.0, "ticks": 10, "weight": 1.0})
        self.current = len(self.steps) - 1
        self.refresh_steps()

    def del_step(self):
        if 0 <= self.current < len(self.steps):
            self.steps.pop(self.current)
            self.current = min(self.current, len(self.steps) - 1)
            self.refresh_steps()

    def move_step(self, delta):
        i = self.current
        j = i + delta
        if 0 <= i < len(self.steps) and 0 <= j < len(self.steps):
            self.steps[i], self.steps[j] = self.steps[j], self.steps[i]
            self.current = j
            self.refresh_steps()

    def update_ticks_label(self):
        total = 0
        parts = []
        for i, st in enumerate(self.steps):
            parts.append(f"{i+1}:{total}-{total + st['ticks'] - 1}")
            total += st["ticks"]
        color = "#ff6b6b" if total < 1 else MUTED
        self.ticks_label.setStyleSheet(f"color:{color}; font-size:12px;")
        self.ticks_label.setText(
            f"时间窗  {'  |  '.join(parts)}\n"
            f"合计 {total} ticks（每次评估就跑 {total} tick）")

    def update_preview(self):
        pm = self._base_pixmap.copy()
        s = self._scale
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#666677"), 1, Qt.DashLine))
        p.drawRect(0, int((422 - 96) * s), int(390 * s), int(192 * s))
        colors = {"move_view": "#7c5cff", "mouse_move": "#3ec6ff", "click": "#5cff9d"}
        for i, st in enumerate(self.steps):
            cur = (i == self.current)
            p.setPen(QPen(QColor(colors[st["type"]]), 3 if cur else 1))
            if st["type"] == "move_view":
                y = st["y"]
                p.drawRect(0, int((y - 96) * s), int(390 * s), int(192 * s))
                p.drawLine(0, int(y * s), int(390 * s), int(y * s))
                p.drawText(6, max(12, int(y * s) - 4), f"{i+1}. view {y:g}")
            else:
                cx, cy, r = st["x"] * s, st["y"] * s, max(3.0, st["tol"] * s)
                p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
                p.drawLine(int(cx - 6), int(cy), int(cx + 6), int(cy))
                p.drawLine(int(cx), int(cy - 6), int(cx), int(cy + 6))
                p.drawText(int(cx + r) + 3, int(cy) - 3, f"{i+1}")
        p.end()
        self.preview.setPixmap(pm)

    def new_lesson(self):
        n, ok = QInputDialog.getInt(self, "新建课程", "课程编号：", 2, 1, 999)
        if not ok:
            return
        name = f"class-{n}.json"
        path = self.class_dir / name
        if path.exists():
            QMessageBox.warning(self, "已存在", f"{name} 已存在。")
            return
        path.write_text(json.dumps(
            [{"type": "move_view", "y": 422, "tol": 40, "ticks": 10, "weight": 1}],
            ensure_ascii=False, indent=4), encoding="utf-8")
        self.refresh_lessons(select=name)

    def delete_lesson(self):
        name = self.lesson_box.currentText()
        if not name:
            return
        if QMessageBox.question(self, "删除课程", f"确定删除 {name}？") != QMessageBox.Yes:
            return
        (self.class_dir / name).unlink(missing_ok=True)
        self.refresh_lessons()

    def save(self):
        name = self.lesson_box.currentText()
        if not name:
            return
        if not self.steps:
            QMessageBox.warning(self, "无法保存", "课程至少需要一个步骤。")
            return
        for i, st in enumerate(self.steps):
            if st["type"] == "move_view" and not (96 <= st["y"] <= 748):
                QMessageBox.warning(self, "参数超出范围",
                                    f"第 {i+1} 步 move_view 的 y 必须在 96~748。")
                return
        path = self.class_dir / name
        ordered = []
        for st in self.steps:
            d = {"type": st["type"]}
            if st["type"] != "move_view":
                d["x"] = st["x"]
            d["y"] = st["y"]
            d["tol"] = st["tol"]
            d["ticks"] = st["ticks"]
            d["weight"] = st["weight"]
            ordered.append(d)
        path.write_text(json.dumps(ordered, ensure_ascii=False, indent=4),
                        encoding="utf-8")
        self.statusBar().showMessage(f"已保存 {path}（下次开始训练时生效）", 8000)


# ===================== 开始训练对话框 =====================

class TrainStartDialog(QDialog):
    """开始训练前的设置：训练代数 + 单个/多个课程选择。"""

    def __init__(self, parent=None, preselect_lessons=None, default_count=10):
        super().__init__(parent)
        self.setWindowTitle("开始训练")
        self.setMinimumWidth(440)
        self._result = None

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(12)

        title = QLabel("训练设置")
        title.setStyleSheet("font-size:16px; font-weight:bold;")
        v.addWidget(title)

        form = QFormLayout()
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(1, 100000)
        self.gen_spin.setValue(max(1, int(default_count)))
        form.addRow("训练多少代：", self.gen_spin)
        v.addLayout(form)

        self.mode_single = QRadioButton("训练单个课程（下拉选一个）")
        self.mode_multi = QRadioButton("训练多个课程（可勾选多个）")
        self.mode_single.setChecked(True)
        v.addWidget(self.mode_single)
        v.addWidget(self.mode_multi)

        self.single_combo = QComboBox()
        v.addWidget(self.single_combo)

        self.multi_list = QListWidget()
        self.multi_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.multi_list.setMaximumHeight(220)
        v.addWidget(self.multi_list)
        self.multi_list.hide()

        self.mode_single.toggled.connect(self._switch_mode)
        self.mode_multi.toggled.connect(self._switch_mode)

        self.hint = QLabel("")
        self.hint.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)

        btns = QHBoxLayout()
        btns.addStretch()
        cancel = QPushButton("取消")
        ok = QPushButton("开始训练")
        ok.setStyleSheet(f"border-color:{ACCENT};")
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self._accept)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        v.addLayout(btns)

        self._fill_lessons(preselect_lessons)
        self._switch_mode()

    def _fill_lessons(self, preselect):
        names = Trainer.available_lessons()
        self.single_combo.clear()
        self.single_combo.addItems(names)
        self.multi_list.clear()
        for nm in names:
            item = QListWidgetItem(nm)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.multi_list.addItem(item)
        if preselect:
            for nm in preselect:
                idx = self.single_combo.findText(nm)
                if idx >= 0:
                    self.single_combo.setCurrentIndex(idx)
                    break
            for j in range(self.multi_list.count()):
                it = self.multi_list.item(j)
                if it.text() in preselect:
                    it.setCheckState(Qt.Checked)
        if not names:
            self.hint.setText("未找到任何课程（data/class/class-*.json），请先在课程编辑器里创建。")

    def _switch_mode(self):
        single = self.mode_single.isChecked()
        self.single_combo.setVisible(single)
        self.multi_list.setVisible(not single)
        total = 0
        if single:
            if self.single_combo.count():
                total = 1
        else:
            for j in range(self.multi_list.count()):
                if self.multi_list.item(j).checkState() == Qt.Checked:
                    total += 1
        if total:
            self.hint.setText(
                f"已选 {total} 个课程。每代每个网络都会跑完所有选中课程，综合得分取平均。")
        else:
            self.hint.setText("请至少选择一个课程。")

    def _selected_lessons(self):
        if self.mode_single.isChecked():
            nm = self.single_combo.currentText()
            return [nm] if nm else []
        out = []
        for j in range(self.multi_list.count()):
            it = self.multi_list.item(j)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    def _accept(self):
        lessons = self._selected_lessons()
        if not lessons:
            QMessageBox.warning(self, "缺少课程", "请至少选择一个课程。")
            return
        if self.mode_single.isChecked() and len(lessons) != 1:
            QMessageBox.warning(self, "单选模式", "单个课程模式只能选择一个课程。")
            return
        self._result = (int(self.gen_spin.value()), lessons)
        self.accept()

    def get_result(self):
        return self._result


# ===================== 控制台主窗口 =====================

class ConsoleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waip AI Q-DRN 控制台")
        self.resize(1100, 720)
        self.setStyleSheet(_style())
        self.trainer = None
        self._replay_windows = []
        self._run_windows = []
        self._log_offset = 0

        # 日志实时尾随：直接读 train.log，避免依赖跨线程信号导致“看不到日志”
        self._log_timer = QTimer(self)
        self._log_timer.setInterval(400)
        self._log_timer.timeout.connect(self._tail_log)

        self._quit_when_done = False
        self._refresh_tick = 0
        self._table_sig = None
        self._pending_meta = None

        # 空转心跳：让主线程定期执行 Python 字节码，
        # 否则 Qt 的 C++ 事件循环里 Ctrl+C(SIGINT) 永远得不到处理
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(100)
        self._pulse_timer.timeout.connect(lambda: None)
        self._pulse_timer.start()
        self._install_sigint()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_mode_page())   # 0
        self.stack.addWidget(self._build_run_page())    # 1
        self.stack.addWidget(self._build_train_page())  # 2
        self.stack.setCurrentIndex(0)
        QTimer.singleShot(0, self._check_lock_hint)

    # ---------- 页面构建 ----------

    def _build_mode_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(60, 60, 60, 60)
        v.setSpacing(30)
        title = QLabel("Waip AI Q-DRN")
        title.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        sub = QLabel("选择模式")
        sub.setStyleSheet(f"color:{MUTED}; font-size:16px;")
        sub.setAlignment(Qt.AlignCenter)
        v.addWidget(title); v.addWidget(sub)

        btns = QHBoxLayout()
        btns.setSpacing(30)
        b_train = QPushButton("🚀  训 练\n\n第一课：精准点击 W 键")
        b_train.setMinimumHeight(180)
        b_train.setStyleSheet(f"font-size:18px; background:#1c1633; border:2px solid {ACCENT};")
        b_train.clicked.connect(self.goto_train)
        b_run = QPushButton("▶  运行模型\n\n打开已有网络，观看实时运行")
        b_run.setMinimumHeight(180)
        b_run.setStyleSheet("font-size:18px; background:#161c22; border:2px solid #2a4a6a;")
        b_run.clicked.connect(self.goto_run)
        btns.addWidget(b_train); btns.addWidget(b_run)
        v.addLayout(btns)
        b_lesson = QPushButton("📝  课程编辑器（可视化编辑 data/class/class-N.json）")
        b_lesson.setStyleSheet("font-size:14px; background:#141a16; border:1px solid #2a4a3a;")
        b_lesson.clicked.connect(self.open_lesson_editor)
        v.addWidget(b_lesson)
        v.addStretch()
        return page

    def open_lesson_editor(self):
        self._lesson_editor = LessonEditorWindow(self)
        self._lesson_editor.show()
        self._lesson_editor.raise_()
        self._lesson_editor.activateWindow()

    def _build_run_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        top = QHBoxLayout()
        back = QPushButton("← 返回"); back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        top.addWidget(back); top.addStretch()
        top.addWidget(QLabel("运行模型"))
        v.addLayout(top)
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QAbstractItemView.SingleSelection)
        v.addWidget(self.run_list, 1)
        row = QHBoxLayout()
        create_btn = QPushButton("新建意识体 (100/3 测试网)")
        create_btn.clicked.connect(self._create_agent)
        start_btn = QPushButton("▶ 启动选中网络")
        start_btn.setStyleSheet(f"border-color:{ACCENT};")
        start_btn.clicked.connect(self._launch_run)
        row.addWidget(create_btn); row.addStretch(); row.addWidget(start_btn)
        v.addLayout(row)
        return page

    def _build_train_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(10, 10, 10, 10)

        top = QHBoxLayout()
        back = QPushButton("← 返回主页"); back.clicked.connect(self._back_from_train)
        top.addWidget(back)
        self.gen_label = QLabel("第 — 代")
        self.gen_label.setStyleSheet("font-size:16px; font-weight:bold;")
        top.addWidget(self.gen_label)
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(f"color:{MUTED};")
        top.addWidget(self.stats_label)
        top.addStretch()
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220); self.progress.setRange(0, 100)
        top.addWidget(self.progress)
        self.train_btn = QPushButton("▶ 开始训练")
        self.train_btn.setStyleSheet(f"border-color:{ACCENT};")
        self.train_btn.clicked.connect(self._toggle_train)
        top.addWidget(self.train_btn)
        v.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        # 左：日志
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(QLabel("训练日志（data/TRAIN/train.log）"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        lv.addWidget(self.log_view, 1)
        lrow = QHBoxLayout()
        reload_log = QPushButton("重新加载日志文件")
        reload_log.clicked.connect(self._reload_log_file)
        lrow.addWidget(reload_log); lrow.addStretch()
        lv.addLayout(lrow)
        split.addWidget(left)
        # 右：网络管理表
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(6, 0, 0, 0)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("当前代网络（点击“观看”查看该网络在课程中的点击过程）"))
        hdr.addStretch()
        hdr.addWidget(QLabel("观看课程"))
        self.watch_lesson_box = QComboBox()
        self.watch_lesson_box.setFixedWidth(120)
        hdr.addWidget(self.watch_lesson_box)
        rv.addLayout(hdr)
        self.net_table = QTableWidget(0, 7)
        self.net_table.setHorizontalHeaderLabels(
            ["编号", "角色", "上一代得分", "命中W", "点击数", "父代", "观看"])
        self.net_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.net_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.net_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        rv.addWidget(self.net_table, 1)
        split.addWidget(right)
        split.setSizes([560, 540])
        v.addWidget(split, 1)
        return page

    # ---------- 页面切换 ----------

    def goto_run(self):
        self._refresh_agent_list()
        self.stack.setCurrentIndex(1)

    def goto_train(self):
        self.stack.setCurrentIndex(2)

        # 已有训练线程在跑 → 回到训练面板，按钮显示“结束训练”
        if self.trainer is not None and self.trainer.is_running():
            self._update_train_button()
            self._refresh_train_view()
            if not self._log_timer.isActive():
                self._log_timer.start()
            return
        # 线程已结束 → 允许重新开始
        if self.trainer is not None and not self.trainer.is_running():
            self.trainer = None

        # 只导航，不弹窗：从磁盘读取上一代元数据，让“观看”在未训练时也能用
        self._load_pending_meta()
        self._reload_log_file()
        self._refresh_train_view()
        self._update_train_button()

        # 检测另一个进程是否正在训练同一份存档
        lock = TrainerMod.read_lock()
        if lock and lock.get("state") == "running":
            pid = lock.get("pid")
            if pid and int(pid) != os.getpid() and TrainerMod.pid_alive(pid):
                self.statusBar().showMessage(
                    f"另一个进程（PID {pid}）正在训练，请先关闭它再开始。")
                return
            # 本进程残留 / 已崩溃的锁：不阻塞，开始训练时会自动从存档继续
            self.statusBar().showMessage(
                f"检测到未完成的训练（第 {lock.get('generation', '?')} 代），"
                "点击「开始训练」将从存档继续。")

    def _load_pending_meta(self):
        path = TRAIN_DIR / "generation.json"
        if path.exists():
            try:
                self._pending_meta = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._pending_meta = None
        else:
            self._pending_meta = None

    def _update_train_button(self):
        running = self.trainer is not None and self.trainer.is_running()
        self.train_btn.setEnabled(True)
        if running:
            self.train_btn.setText("⏹ 结束训练")
        else:
            self.train_btn.setText("▶ 开始训练")

    def _toggle_train(self):
        if self.trainer is not None and self.trainer.is_running():
            self._stop_train()
            return
        # 另一个进程在训练 → 不允许开始
        pre = None
        lock = TrainerMod.read_lock()
        if lock and lock.get("state") == "running":
            pid = lock.get("pid")
            if pid and int(pid) != os.getpid() and TrainerMod.pid_alive(pid):
                QMessageBox.warning(
                    self, "训练进行中",
                    f"检测到另一个进程（PID {pid}）正在训练。\n"
                    "请先关闭那个窗口，避免两个进程同时写同一份存档。")
                return
            pre = lock.get("lessons")
        dlg = TrainStartDialog(self, preselect_lessons=pre)
        if dlg.exec() != QDialog.Accepted:
            return
        count, lessons = dlg.get_result()
        target_gen = self._current_generation() + count
        self._start_train(target_gen=target_gen, lessons=lessons, resumed=False)

    def _current_generation(self):
        path = TRAIN_DIR / "generation.json"
        if path.exists():
            try:
                return int(json.loads(path.read_text(encoding="utf-8")).get("generation", 0))
            except Exception:
                return 0
        return 0

    def _install_sigint(self):
        """Ctrl+C / Ctrl+Break：结束训练并退出控制台。"""
        import signal

        def handler(_signum, _frame):
            try:
                if self.trainer is not None and self.trainer.is_running():
                    if self._quit_when_done:      # 再按一次 → 立即退出
                        QApplication.quit()
                        return
                    self._quit_when_done = True
                    self.trainer.stop()
                    self.statusBar().showMessage(
                        "已收到 Ctrl+C，正在结束训练（本代不会写入数据库）…")
                else:
                    QApplication.quit()
            except Exception:
                pass

        for name in ("SIGINT", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, handler)
            except Exception:
                pass

    def _check_lock_hint(self):
        lock = TrainerMod.read_lock()
        if lock and lock.get("state") == "running":
            self.statusBar().showMessage(
                f"检测到未完成的训练（第 {lock.get('generation', '?')} 代），"
                "点击「训练」将继续。")

    def _back_from_train(self):
        self._log_timer.stop()
        self.stack.setCurrentIndex(0)

    def closeEvent(self, e):
        self._log_timer.stop()
        if self.trainer and self.trainer.is_running():
            self.trainer.stop()
        for w in self._replay_windows:
            try: w.close()
            except Exception: pass
        for w in self._run_windows:
            try: w.close()
            except Exception: pass
        super().closeEvent(e)

    # ---------- 运行模型 ----------

    def _refresh_agent_list(self):
        self.run_list.clear()
        agent_dir = DATA_DIR / "Agent"
        if not agent_dir.exists():
            return
        for name in sorted(os.listdir(agent_dir)):
            sub = agent_dir / name
            if sub.is_dir() and (sub / "Neural Configuration Database.json").exists():
                self.run_list.addItem(QListWidgetItem(name))

    def _create_agent(self):
        import Terminal
        name = Terminal.create_agent()
        if name:
            self._refresh_agent_list()
            QMessageBox.information(self, "新建完成", f"已创建意识体：{name}")

    def _launch_run(self):
        item = self.run_list.currentItem()
        if not item:
            QMessageBox.warning(self, "提示", "请先选择一个意识体。")
            return
        try:
            import Terminal
            from ChatWorkspace import ChatWorkspaceWindow, NeuralWorker
            name = item.text()
            cfg, syn, state = Terminal.load_agent_databases(name)
            sys.setswitchinterval(0.001)
            worker = NeuralWorker(
                cfg, syn, state,
                lambda c, s, sy: Terminal.save_databases(name, c, s, sy),
            )
            win = ChatWorkspaceWindow(name, worker)
            worker.signals.result.connect(win.apply_result)
            worker.signals.error.connect(win.fail)
            worker.start()
            win.show()
            # 控制台已持有唯一的事件循环，这里只需保留窗口引用，不能再调用 app.exec()
            self._run_windows.append(win)
        except Exception as e:
            QMessageBox.critical(self, "运行失败", f"{type(e).__name__}: {e}")
        finally:
            self._refresh_agent_list()

    # ---------- 训练 ----------

    def _start_train(self, target_gen=None, resumed=False, lessons=None):
        try:
            self.statusBar().showMessage("正在准备屏幕（首次会渲染 ChatUI）…")
            QApplication.processEvents()
            screen = Simulator.load_or_render_screen(TRAIN_DIR / "screen.npy")
            cursor_sprite = Simulator.load_or_render_cursor(TRAIN_DIR / "cursor.npz")
        except Exception as e:
            QMessageBox.critical(self, "无法渲染屏幕", str(e))
            self.statusBar().clearMessage()
            return

        self.statusBar().showMessage(
            "继续上次训练…" if resumed else "正在启动训练线程…")
        try:
            sys.setswitchinterval(0.001)   # 训练线程重度占用 GIL 时，保证界面/信号仍能响应
        except Exception:
            pass
        self._quit_when_done = False
        self._table_sig = None
        # 立即读取上一代元数据，让列表在加载存档期间就显示已有的 100 个网络
        self._load_pending_meta()
        try:
            self.trainer = Trainer(screen, target_gen=target_gen,
                                   cursor_sprite=cursor_sprite, lessons=lessons)
            self.trainer.sig_progress.connect(self._on_progress)
            self.trainer.sig_generation_done.connect(self._on_generation_done)
            self.trainer.sig_finished.connect(self._on_finished)
            self._reload_log_file()
            self._log_timer.start()
            self.trainer.start()
        except Exception as e:
            self.trainer = None
            self._log_timer.stop()
            QMessageBox.critical(self, "训练启动失败", f"{type(e).__name__}: {e}")
            self.statusBar().clearMessage()
            self._update_train_button()
            return

        self._update_train_button()
        if target_gen is not None:
            self.statusBar().showMessage(
                f"训练进行中（目标第 {target_gen} 代）· 可按 Ctrl+C 停止")
        else:
            self.statusBar().showMessage("训练进行中（不设上限）· 可按 Ctrl+C 停止")

        # 先把已有的 100 个网络填进表格，不必等全部加载完
        self._table_sig = None
        self._refresh_train_view()

    def _stop_train(self):
        if self.trainer and self.trainer.is_running():
            self.trainer.stop()
            self.statusBar().showMessage("正在结束训练（本代不会写入数据库）…")

    def _on_finished(self):
        self._tail_log()
        self._log_timer.stop()
        self._load_pending_meta()
        self._update_train_button()
        self.statusBar().showMessage("训练已结束，可点击「开始训练」继续。")
        self._refresh_train_view()
        if self._quit_when_done:
            self._quit_when_done = False
            QApplication.quit()

    def _reload_log_file(self):
        path = TRAIN_DIR / "train.log"
        if not path.exists():
            self.log_view.clear()
            self._log_offset = 0
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            self.log_view.setPlainText(text)
            self._log_offset = path.stat().st_size
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception as e:
            self.log_view.appendPlainText(f"[读取日志失败] {e}")

    def _tail_log(self):
        """把 train.log 的新增内容追加到日志视图（兜底，保证实时可见）。"""
        self._refresh_tick += 1
        # 训练中每 ~2s 刷新一次网络表（显示加载进度 / 实时得分）
        if self._refresh_tick % 5 == 0 and self.trainer and self.trainer.is_running():
            self._refresh_train_view()
        path = TRAIN_DIR / "train.log"
        if not path.exists():
            return
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self._log_offset:      # 文件被截断 → 全量重载
            self._reload_log_file()
            return
        if size == self._log_offset:
            return
        try:
            with open(path, "rb") as f:
                f.seek(self._log_offset)
                data = f.read()
                self._log_offset = f.tell()
        except OSError:
            return
        chunk = data.decode("utf-8", errors="replace")
        if chunk:
            self.log_view.appendPlainText(chunk.rstrip("\n"))
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_progress(self, done, total, score):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.stats_label.setText(f"评估中 {done}/{total}  最新得分 {score:+.4f}")

    def _on_generation_done(self, meta):
        self._refresh_train_view()
        self.statusBar().showMessage(
            f"第 {meta['generation']} 代完成：最佳 {meta['prev_best']:+.4f}  "
            f"平均 {meta['prev_mean']:+.4f}  命中W {meta['prev_hit_count']}/100")

    def _refresh_train_view(self):
        trainer = self.trainer
        meta = (trainer.current_meta if trainer is not None else None) or self._pending_meta or {}
        gen = meta.get("generation", (trainer.generation if trainer is not None else 0))
        target = getattr(trainer, "target_gen", None) if trainer is not None else None
        if target is not None:
            self.gen_label.setText(f"第 {gen} 代 / 目标 {target}")
        else:
            self.gen_label.setText(f"第 {gen} 代")
        best = meta.get("prev_best"); mean = meta.get("prev_mean")
        hit = meta.get("prev_hit_count")
        parts = []
        if best is not None: parts.append(f"上一代最佳 {best:+.4f}")
        if mean is not None: parts.append(f"平均 {mean:+.4f}")
        if hit is not None: parts.append(f"命中W {hit}/100")
        boost = meta.get("mutation_boost")
        if boost and boost > 1:
            parts.append(f"趋同 σ={meta.get('score_std', 0):.4f} 变异率×{boost:g}")

        running = trainer is not None and trainer.is_running()
        live = (getattr(trainer, "live_results", None) if trainer is not None else None) or {}
        loaded = len(trainer.population) if trainer is not None else 0

        pop = meta.get("population", [])
        if not pop:
            if running:
                if gen > 0:
                    self.stats_label.setText(f"第 {gen} 代 · 正在加载网络 {loaded}/100…")
                else:
                    self.stats_label.setText(f"首次初始化网络 {loaded}/100…")
            else:
                self.stats_label.setText(
                    "尚未开始训练。点击右上角「开始训练」选择课程与代数；"
                    "下方表格可观看已有回放。")
        elif not (running and live):
            self.stats_label.setText("  |  ".join(parts))

        self._refresh_watch_lessons(meta)

        results = meta.get("results", {})
        if not pop:
            n = loaded
        else:
            n = len(pop)

        # 只在行数变化时重建；其余情况原地改文本，避免每 2s 卡一下界面
        sig = (n, len(live), meta.get("generation"))
        if sig == self._table_sig:
            return
        self._table_sig = sig
        sb = self.net_table.verticalScrollBar()
        scroll_pos = sb.value()
        rebuilt = n != self.net_table.rowCount()
        if rebuilt:
            self.net_table.setRowCount(n)

        for row in range(n):
            m = pop[row] if row < len(pop) else {
                "name": f"net_{row:03d}", "role": None, "parent": None,
                "score": None, "replay_gen": None, "replay_idx": None,
            }
            lr = live.get(row)
            if lr is not None:
                score, res = lr["score"], lr
            elif live:
                # 本代已有实时结果，但该网络还没轮到 → 先显示待评估
                score, res = None, {}
            else:
                score, res = m.get("score"), results.get(m["name"], {})
            items = [
                m["name"],
                {"survivor": "存活", "child": "子代"}.get(m.get("role"), "网络"),
                f"{score:+.4f}" if isinstance(score, (int, float)) else "—",
                "是" if res.get("hit_w") else ("否" if res else "—"),
                str(res.get("clicks", "—")),
                f"net_{m['parent']:03d}" if m.get("parent") is not None else "—",
            ]
            for col, txt in enumerate(items):
                item = self.net_table.item(row, col)
                if item is None:
                    self.net_table.setItem(row, col, QTableWidgetItem(txt))
                elif item.text() != txt:
                    item.setText(txt)

            watch = self.net_table.cellWidget(row, 6)
            if watch is None:
                watch = QPushButton()
                watch.clicked.connect(lambda _=False, r=row: self._watch_row(r))
                self.net_table.setCellWidget(row, 6, watch)
            rg, ri, _, _, _ = self._watch_state(row)
            if rg is None or ri is None:
                if watch.isEnabled() or watch.text() != "未推演":
                    watch.setEnabled(False)
                    watch.setText("未推演")
            else:
                if not watch.isEnabled() or watch.text() != "观看":
                    watch.setEnabled(True)
                    watch.setText("观看")

        if rebuilt:
            self.net_table.resizeRowsToContents()
        sb.setValue(scroll_pos)

    def _refresh_watch_lessons(self, meta):
        lessons = list(meta.get("lessons") or [])
        if not lessons and meta.get("lesson"):
            lessons = [meta["lesson"]]
        current = [self.watch_lesson_box.itemText(i)
                   for i in range(self.watch_lesson_box.count())]
        if current != lessons:
            keep = self.watch_lesson_box.currentText()
            self.watch_lesson_box.blockSignals(True)
            self.watch_lesson_box.clear()
            self.watch_lesson_box.addItems(lessons)
            if keep in lessons:
                self.watch_lesson_box.setCurrentText(keep)
            self.watch_lesson_box.blockSignals(False)
        self.watch_lesson_box.setVisible(len(lessons) > 1)

    def _watch_state(self, row):
        """返回 (replay_gen, replay_idx, score, hit_w, lesson_id)；未推演则前两项为 None。"""
        trainer = self.trainer
        meta = (trainer.current_meta if trainer is not None else None) or self._pending_meta or {}
        lessons = meta.get("lessons") or ([meta["lesson"]] if meta.get("lesson") else [])
        lesson_id = self.watch_lesson_box.currentText() or (lessons[0] if lessons else None)
        live = (getattr(trainer, "live_results", None) if trainer is not None else None) or {}
        lr = live.get(row)
        if lr is not None:
            return (getattr(trainer, "live_gen", None), row,
                    lr["score"], lr["hit_w"], lesson_id)
        if live:
            return None, None, None, None, lesson_id
        pop = meta.get("population") or []
        m = pop[row] if row < len(pop) else {}
        res = (meta.get("results") or {}).get(m.get("name"), {}) or {}
        return m.get("replay_gen"), m.get("replay_idx"), m.get("score"), res.get("hit_w"), lesson_id

    def _watch_row(self, row):
        rg, ri, score, hit, lesson_id = self._watch_state(row)
        if rg is None or ri is None:
            self.statusBar().showMessage("该网络本代还未推演，暂无回放。")
            return
        self._open_replay(rg, ri, f"net_{row:03d}", score, bool(hit), lesson_id)

    def _open_replay(self, gen, idx, name, score, hit_w, lesson_id=None):
        candidates = []
        if lesson_id:
            candidates.append(
                TRAIN_DIR / "replays" / f"gen_{gen}" / lesson_id / f"net_{idx:03d}.json")
        candidates.append(TRAIN_DIR / "replays" / f"gen_{gen}" / f"net_{idx:03d}.json")
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            shown = candidates[0] if candidates else "(无)"
            QMessageBox.warning(self, "无回放", f"找不到回放文件：{shown}")
            return
        try:
            traj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.warning(self, "读取失败", str(e))
            return
        title = f"{name}" + (f" · {lesson_id}" if lesson_id else "")
        w = ReplayWindow(traj, title, score, hit_w, self)
        self._replay_windows.append(w)
        w.show()


def launch():
    app = QApplication.instance() or QApplication(sys.argv)
    win = ConsoleWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch())
