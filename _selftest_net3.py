# -*- coding: utf-8 -*-
"""net-3 闭环自检：仿真引擎 + 合成屏幕 + 模拟 ChatUI 输入法，验证逐键真实点击。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("data").resolve()))
from drn_core import Network, SCREEN_W, SCREEN_H, VIEW_W, VIEW_H, PIXEL_COUNT

ROOT = Path(r"data\Agent\net-3")
cfg = json.loads((ROOT / "Neural Configuration Database.json").read_text(encoding="utf-8"))
syn = json.loads((ROOT / "Synapses Database.json").read_text(encoding="utf-8"))
net = Network(cfg, syn)
print("N =", net.N)

# ---- 合成屏幕：背景灰 15；left_message 时在左侧采样区放亮像素（模拟用户消息）----
BG = 15
def make_screen(left_message=False):
    px = bytearray([BG] * PIXEL_COUNT)
    if left_message:
        # 在 x8..172, y31..190 撒亮像素（文字），足以让 DET 翻转
        for py in range(40, 130, 3):
            for px_ in range(20, 160, 3):
                idx = py * 390 + px_
                px[idx] = 227
    return bytes(px)

# ---- 按键命中矩形（cx,cy,w,h）----
KEY_RECT = {
    'q':(22,694,34,36),'w':(60,694,34,36),'e':(99,694,34,36),'r':(137,694,34,36),
    't':(176,694,34,36),'y':(214,694,34,36),'u':(253,694,34,36),'i':(291,694,34,36),
    'o':(330,694,34,36),'p':(368,694,34,36),
    'a':(24,735,38,36),'s':(67,735,38,36),'d':(109,735,38,36),'f':(152,735,38,36),
    'g':(195,735,38,36),'h':(238,735,38,36),'j':(281,735,38,36),'k':(323,735,38,36),
    'l':(366,735,38,36),
    'z':(67,776,38,36),'x':(109,776,38,36),'c':(152,776,38,36),'v':(195,776,38,36),
    'b':(238,776,38,36),'n':(281,776,38,36),'m':(323,776,38,36),
    '，':(32,817,53,36),'空格':(141,817,155,36),'？':(250,817,53,36),'Enter':(333,817,104,36),
    'send':(355,646,51,34),
}
def hit_key(x, y):
    for label, (cx, cy, w, h) in KEY_RECT.items():
        if abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2:
            return label
    return None

# ---- 模拟 ChatUI 输入法 ----
DICT = {
 'ni':['你','呢','泥','拟','倪'],'hao':['好','号','浩','豪','耗'],
 'dao':['到','道','倒','岛','导'],'di':['底','地','第','低','滴'],
 'bu':['不','部','步','布','补'],'zai':['在','再','载','仔','宰'],
 'ma':['吗','妈','码','马','骂'],'wo':['我','窝','握','卧','沃'],
 'shi':['是','时','事','十','史'],'de':['的','得','地','德','底'],
 'hui':['会','回','汇','辉','绘'],'yi':['一','以','已','意','依'],
}
class IME:
    def __init__(self):
        self.mode='zh'; self.buffer=''; self.committed=''; self.sent=[]
    def press(self, k):
        if k=='中/英': self.mode='en' if self.mode=='zh' else 'zh'; return
        if k=='⌫':
            if self.buffer: self.buffer=self.buffer[:-1]
            elif self.committed: self.committed=self.committed[:-1]
            return
        if k=='空格':
            if self.buffer and self.buffer in DICT:
                self.committed += DICT[self.buffer][0]; self.buffer=''
            else:
                self.committed += ' '
            return
        if k=='Enter' or k=='send':
            self.send(); return
        if k=='，' or k=='？':
            self.committed += k; return
        if self.mode=='en':
            self.committed += k; return
        self.buffer += k
    def send(self):
        txt = self.committed + self.buffer
        if txt:
            self.sent.append(txt)
        self.committed=''; self.buffer=''

MSG1 = ['n','i','空格','h','a','o','空格','，','n','i','空格','z','a','i','空格','m','a','空格','？','send']
MSG2 = ['n','i','空格','d','a','o','空格','d','i','空格','z','a','i','空格','b','u','空格','z','a','i','空格','？','send']
EXPECTED = MSG1 + MSG2

def run_case(name, left_at_start, reply_after_msg1):
    net.reset_state()
    ime = IME()
    cursor_x, cursor_y, view_y = 195.0, 422.0, 422.0
    clicks = []   # (tick, x, y, hit)
    msg1_sent_tick = None
    n_steps = 380
    for t in range(n_steps):
        left = left_at_start
        if reply_after_msg1 and msg1_sent_tick is not None and t > msg1_sent_tick + 1:
            left = True
        px = make_screen(left_message=left)
        net.inject(px, cursor_x, cursor_y, view_y)
        net.propagate()
        outs = net.read_outputs()
        cursor_x = max(0.0, min(390.0, cursor_x + 390.0 * outs[0]))
        cursor_y = max(0.0, min(844.0, cursor_y + 844.0 * outs[1]))
        view_y = max(96.0, min(748.0, view_y + 844.0 * outs[2]))
        if outs[3] > 0:
            hit = hit_key(cursor_x, cursor_y)
            clicks.append((t, round(cursor_x,1), round(cursor_y,1), hit))
            if hit:
                ime.press(hit)
                if hit in ('send','Enter') and msg1_sent_tick is None and len(ime.sent)>=1:
                    msg1_sent_tick = t
    # 报告
    hit_labels = [c[3] for c in clicks]
    ok_text = (ime.sent == ["你好，你在吗？","你到底在不在？"]) if not (left_at_start or reply_after_msg1) else True
    print(f"\n=== {name} ===")
    print(f"  clicks={len(clicks)}  sent={ime.sent}")
    # 检查每个点击命中是否符合预期序列（仅当应该发消息时）
    # 左侧已有消息 → 一次都不点；先发第一条后有回复 → 只点 MSG1；否则两条全点。
    if left_at_start:
        expected_clicks = []
    elif reply_after_msg1:
        expected_clicks = MSG1
    else:
        expected_clicks = EXPECTED
    mismatches = []
    if len(hit_labels) != len(expected_clicks):
        mismatches.append(("COUNT", len(expected_clicks), len(hit_labels), None))
    for i, (got, exp) in enumerate(zip(hit_labels, expected_clicks)):
        if got != exp:
            mismatches.append((i, exp, got, clicks[i]))
    if mismatches:
        print("  MISMATCHES (idx, expected, got, click):")
        for m in mismatches[:20]:
            print("   ", m)
    elif expected_clicks:
        print(f"  all {len(expected_clicks)} clicks hit the correct key ✓")
    return clicks, ime.sent, mismatches

run_case("空聊天（应发两条）", left_at_start=False, reply_after_msg1=False)
run_case("左侧已有用户消息（应不发）", left_at_start=True, reply_after_msg1=False)
run_case("发完第一条后用户回复（应不发第二条）", left_at_start=False, reply_after_msg1=True)
