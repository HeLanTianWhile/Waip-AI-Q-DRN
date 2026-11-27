# -*- coding: utf-8 -*-
"""探针：盯第一个键槽的光标伺服，找出 (target - cursor) 恒定偏移的来源。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("data").resolve()))
from drn_core import Network, PIXEL_COUNT

ROOT = Path(r"data\Agent\net-3")
cfg = json.loads((ROOT / "Neural Configuration Database.json").read_text(encoding="utf-8"))
syn = json.loads((ROOT / "Synapses Database.json").read_text(encoding="utf-8"))
net = Network(cfg, syn)

BG = 15
def screen():
    return bytes([BG] * PIXEL_COUNT)

# 槽时序（与 _build_net3 一致）
KEYS = {
 'q':(22,694),'w':(60,694),'e':(99,694),'r':(137,694),'t':(176,694),'y':(214,694),
 'u':(253,694),'i':(291,694),'o':(330,694),'p':(368,694),
 'a':(24,735),'s':(67,735),'d':(109,735),'f':(152,735),'g':(195,735),'h':(238,735),
 'j':(281,735),'k':(323,735),'l':(366,735),
 'z':(67,776),'x':(109,776),'c':(152,776),'v':(195,776),'b':(238,776),'n':(281,776),'m':(323,776),
 '，':(32,817),'空格':(141,817),'？':(250,817),'Enter':(333,817),'send':(355,646),
}
MSG1 = ['n','i','空格','h','a','o','空格','，','n','i','空格','z','a','i','空格','m','a','空格','？','send']
MSG2 = ['n','i','空格','d','a','o','空格','d','i','空格','z','a','i','空格','b','u','空格','z','a','i','空格','？','send']
L_FIRST=9; L_KEY=7
slots=[]
slots.append(("idle",KEYS[MSG1[0]],8))
for i,k in enumerate(MSG1): slots.append(("key",KEYS[k], L_FIRST if i==0 else L_KEY))
slots.append(("idle",(166,646),22))
for i,k in enumerate(MSG2): slots.append(("key",KEYS[k], L_FIRST if i==0 else L_KEY))
pos=0; timed=[]
for kind,tgt,L in slots:
    timed.append((kind,tgt,pos,pos+L)); pos+=L

def active_target(t):
    for kind,tgt,a,b in timed:
        if a <= t < b: return tgt
    return None

net.reset_state()
cx,cy,vy=195.0,422.0,422.0
print("tick  cursor_x cursor_y | out0   out1   | target    dx     dy")
for t in range(30):
    net.inject(screen(), cx, cy, vy)
    net.propagate()
    o=net.read_outputs()
    tgt=active_target(t)
    if tgt:
        print(f"{t:3d}  {cx:7.1f}  {cy:7.1f} | {o[0]:+.3f} {o[1]:+.3f} | ({tgt[0]:3d},{tgt[1]:3d}) {cx-tgt[0]:+6.1f} {cy-tgt[1]:+6.1f}")
    cx=max(0.0,min(390.0,cx+390.0*o[0]))
    cy=max(0.0,min(844.0,cy+844.0*o[1]))
    vy=max(96.0,min(748.0,vy+844.0*o[2]))
