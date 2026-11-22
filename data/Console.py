# -*- coding: utf-8 -*-
"""
Waip AI Q-DRN 控制台（启动弹窗）。

终端启动后不再直接运行，而是弹出本控制台，提供两个入口：
  · 训练 —— 启动 GA 训练（第一课：点击 W 键），面板内可实时查看训练日志、
            管理当前代网络、并“观看”任一存活网络的 50 tick 点击回放。
  · 运行模型 —— 选择一个已训练的意识体，以旧方式打开 ChatUI 工作区观看运行。
"""

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QStackedWidget,
    QSplitter, QAbstractItemView, QMessageBox, QComboBox, QInputDialog,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

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
    """播放某个网络 50 tick 的点击回放。"""

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
        v.addStretch()
        return page

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
        self.stop_btn = QPushButton("⏹ 结束训练")
        self.stop_btn.clicked.connect(self._stop_train)
        top.addWidget(self.stop_btn)
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
        rv.addWidget(QLabel("当前代网络（点击“观看”查看该网络上一代如何点击 W 键）"))
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

        # 已有训练线程在跑 → 直接回到训练面板
        if self.trainer is not None and self.trainer.is_running():
            self._refresh_train_view()
            return
        # 线程已结束 → 允许重新开始
        if self.trainer is not None and not self.trainer.is_running():
            self.trainer = None

        self._reload_log_file()

        # 1) 检查训练标记：有 → 继续上次训练
        lock = TrainerMod.read_lock()
        if lock and lock.get("state") == "running":
            pid = lock.get("pid")
            if pid and int(pid) != os.getpid() and TrainerMod.pid_alive(pid):
                QMessageBox.warning(
                    self, "训练进行中",
                    f"检测到另一个进程（PID {pid}）正在训练。\n"
                    "请先关闭那个窗口，避免两个进程同时写同一份存档。")
                return
            target_gen = lock.get("target_gen")
            target_gen = int(target_gen) if target_gen is not None else None
            self._start_train(target_gen=target_gen, resumed=True)
            return

        # 2) 没有标记 → 询问训练范围
        count = self._ask_train_scope()
        if count is None:
            self.stack.setCurrentIndex(0)
            return
        target_gen = self._current_generation() + count
        self._start_train(target_gen=target_gen, resumed=False)

    def _current_generation(self):
        path = TRAIN_DIR / "generation.json"
        if path.exists():
            try:
                return int(json.loads(path.read_text(encoding="utf-8")).get("generation", 0))
            except Exception:
                return 0
        return 0

    def _ask_train_scope(self):
        """返回本次要训练的代数；取消返回 None。"""
        box = QMessageBox(self)
        box.setWindowTitle("训练设置")
        box.setIcon(QMessageBox.Question)
        box.setText("本次训练要跑多少代？")
        one_btn = box.addButton("训练一代", QMessageBox.AcceptRole)
        many_btn = box.addButton("指定代数…", QMessageBox.ActionRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is one_btn:
            return 1
        if clicked is many_btn:
            n, ok = QInputDialog.getInt(
                self, "训练代数", "本次训练多少代？", 10, 1, 100000)
            return n if ok else None
        return None

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

    def _start_train(self, target_gen=None, resumed=False):
        try:
            self.statusBar().showMessage("正在准备屏幕（首次会渲染 ChatUI）…")
            QApplication.processEvents()
            screen = Simulator.load_or_render_screen(TRAIN_DIR / "screen.npy")
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
        self._pending_meta = None
        meta_path = TRAIN_DIR / "generation.json"
        if meta_path.exists():
            try:
                self._pending_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self._pending_meta = None
        try:
            self.trainer = Trainer(screen, target_gen=target_gen)
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
            return

        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("⏹ 结束训练")
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
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("训练已结束。")
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
        if not self.trainer:
            return
        meta = self.trainer.current_meta or self._pending_meta or {}
        gen = meta.get("generation", self.trainer.generation)
        if getattr(self.trainer, "target_gen", None) is not None:
            self.gen_label.setText(f"第 {gen} 代 / 目标 {self.trainer.target_gen}")
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
        if not (self.trainer.is_running()
                and getattr(self.trainer, "live_results", None)):
            self.stats_label.setText("  |  ".join(parts))

        pop = meta.get("population", [])
        results = meta.get("results", {})
        live = getattr(self.trainer, "live_results", None) or {}
        loaded = len(self.trainer.population)

        if not pop:
            # 没有任何一代元数据：按已加载数量显示
            if self.trainer.is_running():
                if gen > 0:
                    self.stats_label.setText(f"第 {gen} 代 · 正在加载网络 {loaded}/100…")
                else:
                    self.stats_label.setText(f"首次初始化网络 {loaded}/100…")
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
            rg, ri, _, _ = self._watch_state(row)
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

    def _watch_state(self, row):
        """返回 (replay_gen, replay_idx, score, hit_w)；本代未推演则为 None。"""
        meta = self.trainer.current_meta or self._pending_meta or {}
        live = getattr(self.trainer, "live_results", None) or {}
        lr = live.get(row)
        if lr is not None:
            return getattr(self.trainer, "live_gen", None), row, lr["score"], lr["hit_w"]
        if live:
            return None, None, None, None
        pop = meta.get("population") or []
        m = pop[row] if row < len(pop) else {}
        res = (meta.get("results") or {}).get(m.get("name"), {}) or {}
        return m.get("replay_gen"), m.get("replay_idx"), m.get("score"), res.get("hit_w")

    def _watch_row(self, row):
        rg, ri, score, hit = self._watch_state(row)
        if rg is None or ri is None:
            self.statusBar().showMessage("该网络本代还未推演，暂无回放。")
            return
        self._open_replay(rg, ri, f"net_{row:03d}", score, bool(hit))

    def _open_replay(self, gen, idx, name, score, hit_w):
        path = TRAIN_DIR / "replays" / f"gen_{gen}" / f"net_{idx:03d}.json"
        if not path.exists():
            QMessageBox.warning(self, "无回放", f"找不到回放文件：{path}")
            return
        try:
            traj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.warning(self, "读取失败", str(e))
            return
        w = ReplayWindow(traj, name, score, hit_w, self)
        self._replay_windows.append(w)
        w.show()


def launch():
    app = QApplication.instance() or QApplication(sys.argv)
    win = ConsoleWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch())
