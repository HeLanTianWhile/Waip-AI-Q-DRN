# -*- coding: utf-8 -*-
"""离屏渲染 ChatUI，量出每个按键/发送按钮/输入框在 #app 坐标系下的中心坐标。"""
import os
import json
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QUrl, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView

HTML = Path(r"ChatUI\index.html").resolve()

app = QApplication.instance() or QApplication([])
view = QWebEngineView()
view.resize(390, 844)
loop = QEventLoop()

def on_loaded(ok):
    if not ok:
        print("LOAD FAILED"); loop.quit(); return
    # 等一帧让布局稳定
    QTimer.singleShot(300, collect)

def collect():
    js = r"""
    (function(){
      const app = document.getElementById('app');
      const ar = app.getBoundingClientRect();
      function rect(el){
        const r = el.getBoundingClientRect();
        return {
          label: el.dataset && el.dataset.k ? el.dataset.k : (el.id || el.className),
          cx: Math.round(r.left - ar.left + r.width/2),
          cy: Math.round(r.top - ar.top + r.height/2),
          x: Math.round(r.left - ar.left), y: Math.round(r.top - ar.top),
          w: Math.round(r.width), h: Math.round(r.height)
        };
      }
      const out = {};
      out.keys = [...document.querySelectorAll('.key')].map(rect);
      out.send = rect(document.getElementById('send'));
      out.field = rect(document.getElementById('field'));
      out.composer = rect(document.querySelector('.composer'));
      out.candbar = rect(document.querySelector('.candbar'));
      out.kb = rect(document.getElementById('kb'));
      out.log = rect(document.getElementById('log'));
      return JSON.stringify(out);
    })();
    """
    view.page().runJavaScript(js, on_result)

def on_result(s):
    print(s)
    loop.quit()

view.loadFinished.connect(on_loaded)
view.load(QUrl.fromLocalFile(str(HTML)))
loop.exec()
