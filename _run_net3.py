# -*- coding: utf-8 -*-
"""临时启动器：直接以 net-3 打开 ChatUI 工作区，用于真机验证。

不落盘（save_callback 为 no-op），避免运行时状态覆盖 data/Agent/net-3 的原始数据库。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("data").resolve()))

import Terminal
from PySide6.QtWidgets import QApplication
from ChatWorkspace import ChatWorkspaceWindow, NeuralWorker

NAME = "net-3"
cfg, syn, state = Terminal.load_agent_databases(NAME)
print(f"[run_net3] 载入 {NAME}: 神经元 {len(cfg)}  突触 {len(syn)}", flush=True)

app = QApplication(sys.argv)
sys.setswitchinterval(0.001)
worker = NeuralWorker(cfg, syn, state, lambda c, s, sy: None)
win = ChatWorkspaceWindow(NAME, worker)
worker.signals.result.connect(win.apply_result)
worker.signals.error.connect(win.fail)
worker.start()
win.show()
sys.exit(app.exec())
