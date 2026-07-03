# -*- coding: utf-8 -*-
"""
悬浮 AI 对话球 (RK3588 solder UI) — iOS 风
-------------------------------------------------
悬浮球(可拖动) -> 点击展开半透明可拖对话框，支持：
  · 文字输入 + 发送
  · 语音输入(板载 arecord 录音 -> PC ASR 识别 -> 填入输入框)
  · 暂停按钮 (/stop 中断当前任务)
  · 新对话按钮 (/new 掐断记忆，和 QQ 机器人同一套)
  · 直接与 Intel 端 GenericAgent 对话(带会话记忆)

后端：
  ASR : POST 原始wav字节 -> http://192.168.137.222:8010/asr    -> {ok,text}
  CHAT: POST json{text,session,token} -> http://192.168.137.222:8930/chat -> {ok,reply,session}

最小侵入：主程序 win.show() 后调用 attach_voice_button(win) 即可。
"""
import os, json, urllib.request
from PyQt5.QtCore import Qt, QProcess, QThread, pyqtSignal, QTimer, QPoint
from PyQt5.QtWidgets import (
    QPushButton, QLineEdit, QTextEdit, QPlainTextEdit, QApplication, QWidget,
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QScroller, QScrollerProperties,
    QAbstractScrollArea, QScrollArea, QSizePolicy
)
from PyQt5.QtGui import QFont, QFontDatabase

# ---- 后端地址 ----
PC_HOST   = "192.168.137.222"
ASR_URL   = "http://%s:8010/asr"  % PC_HOST
CHAT_URL  = "http://%s:8930/chat" % PC_HOST
STREAM_URL = "http://%s:8930/chat_stream" % PC_HOST
CHAT_TOKEN = "edispense2026"
CHAT_SESSION = "board"           # 固定会话，保持上下文记忆
# ---- 录音 ----
ARECORD_DEV = "hw:1,0"
WAV_PATH    = "/tmp/voice_input.wav"

# ---- iOS 风字体栈（板子已装 PingFang SC） ----
FONT_FAMILY = '"PingFang SC","PingFang HK","Source Han Sans SC","Noto Sans CJK SC",sans-serif'

# ---- 标题手写马克笔字体（随包 ttf，运行时加载）----
_HERE = os.path.dirname(os.path.abspath(__file__))
TITLE_FONT_PATH = os.path.join(_HERE, "fonts", "Aidian-Marker.ttf")
# 加载失败时的回退族名
TITLE_FONT_FAMILY = "Aidian SignatureTi Marker pen Medium"

def _load_title_font():
    """把随包 ttf 注册到 Qt 字体库，返回真实族名（需 QApplication 已创建）"""
    global TITLE_FONT_FAMILY
    try:
        if not os.path.exists(TITLE_FONT_PATH):
            return
        fid = QFontDatabase.addApplicationFont(TITLE_FONT_PATH)
        if fid < 0:
            return
        fams = QFontDatabase.applicationFontFamilies(fid)
        if fams:
            TITLE_FONT_FAMILY = fams[0]
    except Exception:
        pass


def _apply_smooth_scroll(area):
    """给滚动区域(QTextEdit/QAbstractScrollArea)加触摸动量滚动，丝滑效果"""
    try:
        QScroller.grabGesture(area.viewport(), QScroller.LeftMouseButtonGesture)
        sc = QScroller.scroller(area.viewport())
        props = sc.scrollerProperties()
        props.setScrollMetric(QScrollerProperties.DragVelocitySmoothingFactor, 0.6)
        props.setScrollMetric(QScrollerProperties.DecelerationFactor, 0.35)
        props.setScrollMetric(QScrollerProperties.MinimumVelocity, 0.0)
        props.setScrollMetric(QScrollerProperties.MaximumVelocity, 0.6)
        props.setScrollMetric(QScrollerProperties.AxisLockThreshold, 0.66)
        props.setScrollMetric(QScrollerProperties.OvershootDragResistanceFactor, 0.33)
        props.setScrollMetric(QScrollerProperties.OvershootScrollDistanceFactor, 0.12)
        props.setScrollMetric(QScrollerProperties.OvershootDragDistanceFactor, 0.12)
        props.setScrollMetric(QScrollerProperties.SnapPositionRatio, 0.5)
        sc.setScrollerProperties(props)
    except Exception:
        pass


# ============================================================
#  后台线程
# ============================================================
class _RecognizeThread(QThread):
    """把 wav POST 到 PC ASR 服务，不阻塞 UI"""
    done = pyqtSignal(str)
    fail = pyqtSignal(str)

    def __init__(self, wav_path):
        super().__init__()
        self.wav_path = wav_path

    def run(self):
        try:
            with open(self.wav_path, "rb") as f:
                data = f.read()
            if len(data) < 100:
                self.fail.emit("录音为空")
                return
            req = urllib.request.Request(
                ASR_URL, data=data,
                headers={"Content-Type": "application/octet-stream"},
                method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            if obj.get("ok"):
                self.done.emit(obj.get("text", "").strip())
            else:
                self.fail.emit(str(obj.get("error", "识别失败")))
        except Exception as e:
            self.fail.emit(str(e))


class _ChatThread(QThread):
    """把用户输入 POST 到 GenericAgent，不阻塞 UI。text 可为普通消息或 /stop /new"""
    done = pyqtSignal(str)
    fail = pyqtSignal(str)

    def __init__(self, text, timeout=300):
        super().__init__()
        self.text = text
        self.timeout = timeout

    def run(self):
        try:
            payload = json.dumps({
                "text": self.text,
                "session": CHAT_SESSION,
                "token": CHAT_TOKEN,
            }).encode("utf-8")
            req = urllib.request.Request(
                CHAT_URL, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            if obj.get("ok"):
                self.done.emit(obj.get("reply", "").strip() or "(空回复)")
            else:
                self.fail.emit(str(obj.get("error", "对话失败")))
        except Exception as e:
            self.fail.emit(str(e))


class _StreamChatThread(QThread):
    """流式：POST /chat_stream，逐行读取 NDJSON，每个 turn emit 一条信号。
    turn(turn_no, text) 每个思考 turn 一个气泡；done(reply) 最终答复；fail(err)。"""
    turn = pyqtSignal(int, str)
    done = pyqtSignal(str)
    fail = pyqtSignal(str)

    def __init__(self, text, timeout=600):
        super().__init__()
        self.text = text
        self.timeout = timeout

    def run(self):
        try:
            payload = json.dumps({
                "text": self.text,
                "session": CHAT_SESSION,
                "token": CHAT_TOKEN,
            }).encode("utf-8")
            req = urllib.request.Request(
                STREAM_URL, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:  # 按行迭代，NDJSON 每行一个事件
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    t = obj.get("type")
                    if t == "turn":
                        self.turn.emit(int(obj.get("turn", 0)),
                                       (obj.get("text") or "").strip())
                    elif t == "done":
                        self.done.emit((obj.get("reply") or "").strip() or "(空回复)")
                        return
                    elif t == "error":
                        self.fail.emit(str(obj.get("error", "对话失败")))
                        return
            self.done.emit("")  # 流结束却没收到 done
        except Exception as e:
            self.fail.emit(str(e))


# ============================================================
#  半透明对话面板
# ============================================================
class ChatPanel(QFrame):
    """半透明可拖动对话框：标题栏(新对话/暂停/收起) + 消息区 + 输入 + 语音 + 发送"""

    def __init__(self, parent, ball):
        super().__init__(parent)
        self._ball = ball
        self._rec_proc = None
        self._voice_state = 0        # 0 idle, 1 recording, 2 processing
        self._last_edit = None       # 语音识别回填目标（这里固定输入框）
        self._ct = None              # 当前对话线程
        self._st = None              # /stop 线程
        self._nt = None              # /new 线程
        self.setFixedSize(600, 860)
        self.setStyleSheet(
            "ChatPanel{background:rgba(242,242,247,0.80);border-radius:26px;"
            "border:1px solid rgba(255,255,255,0.35);"
            "font-family:%s;}" % FONT_FAMILY)
        self._build_ui()
        # 拖动
        self._drag = False
        self._press = None
        self._start = None

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(10)

        # 标题栏(可拖动区) + 新对话 + 暂停 + 收起
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addStretch(1)                       # 左弹簧：让标题稍微往中间挪
        self._title = QLabel("EDispense AI")
        self._title.setStyleSheet(
            "color:#1c1c1e;font-size:28px;font-weight:500;letter-spacing:1px;"
            "font-family:'%s';" % TITLE_FONT_FAMILY)
        top.addWidget(self._title)
        top.addStretch(1)                       # 右弹簧：与左弹簧对称，标题居中于按钮左侧区

        # 新对话按钮
        self._btn_new = QPushButton("＋")
        self._btn_new.setFixedSize(54, 54)
        self._btn_new.setToolTip("新对话")
        self._btn_new.setFocusPolicy(Qt.NoFocus)
        self._btn_new.setStyleSheet(
            "QPushButton{color:#0a84ff;background:rgba(10,132,255,0.12);"
            "border:none;border-radius:27px;font-size:30px;font-weight:500;}"
            "QPushButton:pressed{background:rgba(10,132,255,0.28);}")
        self._btn_new.clicked.connect(self._on_new)
        top.addWidget(self._btn_new)

        # 暂停按钮
        self._btn_stop = QPushButton("⏸")
        self._btn_stop.setFixedSize(54, 54)
        self._btn_stop.setToolTip("暂停")
        self._btn_stop.setFocusPolicy(Qt.NoFocus)
        self._btn_stop.setStyleSheet(
            "QPushButton{color:#ff9500;background:rgba(255,149,0,0.14);"
            "border:none;border-radius:27px;font-size:24px;}"
            "QPushButton:pressed{background:rgba(255,149,0,0.30);}")
        self._btn_stop.clicked.connect(self._on_stop)
        top.addWidget(self._btn_stop)

        # 收起按钮
        btn_min = QPushButton("—")
        btn_min.setFixedSize(54, 54)
        btn_min.setStyleSheet(
            "QPushButton{color:#8e8e93;background:rgba(120,120,128,0.12);"
            "border:none;border-radius:27px;font-size:28px;}"
            "QPushButton:pressed{background:rgba(120,120,128,0.28);}")
        btn_min.setFocusPolicy(Qt.NoFocus)
        btn_min.clicked.connect(self.collapse)
        top.addWidget(btn_min)
        root.addLayout(top)

        # 消息区（QScrollArea + 气泡容器，QLabel 才能渲染真圆角气泡）
        self._msgs = QScrollArea()
        self._msgs.setWidgetResizable(True)
        self._msgs.setFrameShape(QFrame.NoFrame)
        self._msgs.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._msgs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._msgs.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:6px;background:transparent;margin:4px 2px;}"
            "QScrollBar::handle:vertical{background:rgba(0,0,0,0.20);"
            "border-radius:3px;min-height:36px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}")

        self._msg_wrap = QWidget()
        self._msg_wrap.setStyleSheet("background:transparent;")
        self._msg_col = QVBoxLayout(self._msg_wrap)
        self._msg_col.setContentsMargins(4, 6, 4, 6)
        self._msg_col.setSpacing(0)
        self._msg_col.addStretch(1)   # 底部弹簧，气泡从上往下堆叠
        self._msgs.setWidget(self._msg_wrap)

        _apply_smooth_scroll(self._msgs)
        root.addWidget(self._msgs, 1)

        # 输入行：语音 + 文本框 + 发送
        row = QHBoxLayout()
        row.setSpacing(10)
        self._btn_voice = QPushButton("🎤")
        self._btn_voice.setFixedSize(62, 62)
        self._btn_voice.setFocusPolicy(Qt.NoFocus)
        self._btn_voice.clicked.connect(self._on_voice)
        self._style_voice()
        row.addWidget(self._btn_voice)

        self._input = QLineEdit()
        self._input.setPlaceholderText("说点什么…")
        self._input.setStyleSheet(
            "QLineEdit{background:#ffffff;color:#1c1c1e;"
            "border:1px solid rgba(0,0,0,0.10);border-radius:20px;"
            "font-size:24px;padding:0 20px;"
            "font-family:%s;}"
            "QLineEdit:focus{border:1px solid #0a84ff;}" % FONT_FAMILY)
        self._input.setMinimumHeight(62)
        self._input.returnPressed.connect(self._on_send)
        self._last_edit = self._input
        row.addWidget(self._input, 1)

        self._btn_send = QPushButton("发送")
        self._btn_send.setFixedSize(96, 62)
        self._btn_send.setFocusPolicy(Qt.NoFocus)
        self._btn_send.setStyleSheet(
            "QPushButton{background:#0a84ff;color:white;border:none;"
            "border-radius:20px;font-size:24px;font-weight:600;"
            "font-family:%s;}"
            "QPushButton:pressed{background:#0060df;}"
            "QPushButton:disabled{background:#c7c7cc;color:#ffffff;}"
            % FONT_FAMILY)
        self._btn_send.clicked.connect(self._on_send)
        row.addWidget(self._btn_send)
        root.addLayout(row)

        # 录音闪烁
        self._blink = False
        self._btimer = QTimer(self)
        self._btimer.setInterval(500)
        self._btimer.timeout.connect(self._toggle_blink)

        self._add_msg("assistant", "你好，我是 EDispense-AI，有什么可以帮你？")

    # ---------- 消息渲染（QLabel 气泡，真圆角） ----------
    def _add_msg(self, role, text):
        text = (text or "").strip()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setMaximumWidth(440)
        bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        if role == "user":
            bubble.setStyleSheet(
                "QLabel{background:#0a84ff;color:#ffffff;"
                "border-radius:22px;padding:14px 20px;"
                "font-size:23px;line-height:150%%;font-family:%s;}" % FONT_FAMILY)
            row.addStretch(1)
            row.addWidget(bubble)
        elif role == "system":
            bubble.setMaximumWidth(500)
            bubble.setAlignment(Qt.AlignCenter)
            bubble.setStyleSheet(
                "QLabel{background:rgba(120,120,128,0.16);color:#8e8e93;"
                "border-radius:16px;padding:8px 18px;"
                "font-size:18px;font-family:%s;}" % FONT_FAMILY)
            row.addStretch(1)
            row.addWidget(bubble)
            row.addStretch(1)
        else:
            bubble.setStyleSheet(
                "QLabel{background:#ffffff;color:#1c1c1e;"
                "border-radius:22px;padding:14px 20px;"
                "font-size:23px;line-height:150%%;font-family:%s;}" % FONT_FAMILY)
            row.addWidget(bubble)
            row.addStretch(1)

        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        holder.setContentsMargins(0, 0, 0, 0)
        vb = QVBoxLayout(holder)
        vb.setContentsMargins(4, 6, 4, 6)
        vb.addLayout(row)

        # 插到底部弹簧之前
        self._msg_col.insertWidget(self._msg_col.count() - 1, holder)
        QTimer.singleShot(30, self._scroll_bottom)

    def _scroll_bottom(self):
        sb = self._msgs.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_msgs(self):
        # 移除底部弹簧之前的所有气泡
        while self._msg_col.count() > 1:
            item = self._msg_col.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    # ---------- 发送 ----------
    def _on_send(self):
        text = self._input.text().strip()
        if not text:
            return
        self._add_msg("user", text)
        self._input.clear()
        self._set_busy(True)
        self._ct = _StreamChatThread(text)
        self._ct.turn.connect(self._on_turn)
        self._ct.done.connect(self._on_reply)
        self._ct.fail.connect(self._on_reply_fail)
        self._ct.start()

    def _on_turn(self, turn_no, text):
        # 每个思考 turn 一个气泡：只推纯结果，不带 "Turn N" 标签
        if not text:
            return
        self._add_msg("assistant", text)

    def _on_reply(self, reply):
        # 最后一个 turn 已经通过 turn 信号推过气泡了，done 只负责收尾，
        # 不再重复 add_msg（否则末条消息会出现两遍）。
        self._set_busy(False)

    def _on_reply_fail(self, msg):
        self._set_busy(False)
        self._add_msg("assistant", "出错了：%s" % msg)

    def _set_busy(self, busy):
        self._btn_send.setEnabled(not busy)
        self._btn_send.setText("…" if busy else "发送")

    # ---------- 暂停 /stop ----------
    def _on_stop(self):
        # 不阻塞，独立短超时线程发 /stop（后端绕锁直接 abort）
        self._st = _ChatThread("/stop", timeout=15)
        self._st.done.connect(self._on_stop_done)
        self._st.fail.connect(lambda m: self._add_msg("system", "暂停失败：%s" % m))
        self._st.start()

    def _on_stop_done(self, reply):
        self._set_busy(False)
        self._add_msg("system", reply or "已中断")

    # ---------- 新对话 /new ----------
    def _on_new(self):
        self._nt = _ChatThread("/new", timeout=15)
        self._nt.done.connect(self._on_new_done)
        self._nt.fail.connect(lambda m: self._add_msg("system", "新对话失败：%s" % m))
        self._nt.start()

    def _on_new_done(self, reply):
        self._set_busy(False)
        self._clear_msgs()
        self._add_msg("system", reply or "已开启新对话")
        self._add_msg("assistant", "你好，我是 EDispense-AI，有什么可以帮你？")

    # ---------- 语音 ----------
    def _style_voice(self):
        if self._voice_state == 1:
            bg = "#ff453a" if not getattr(self, "_blink", False) else "#ff8078"
            txt = "⏹"
            fg = "white"
        elif self._voice_state == 2:
            bg, txt, fg = "#c7c7cc", "…", "white"
        else:
            bg, txt, fg = "rgba(10,132,255,0.12)", "🎤", "#0a84ff"
        self._btn_voice.setText(txt)
        self._btn_voice.setStyleSheet(
            "QPushButton{background:%s;color:%s;border:none;"
            "border-radius:31px;font-size:26px;}" % (bg, fg))

    def _toggle_blink(self):
        self._blink = not self._blink
        self._style_voice()

    def _on_voice(self):
        if self._voice_state == 0:
            self._start_record()
        elif self._voice_state == 1:
            self._stop_and_recognize()

    def _start_record(self):
        try:
            if os.path.exists(WAV_PATH):
                os.remove(WAV_PATH)
        except Exception:
            pass
        self._rec_proc = QProcess(self)
        self._rec_proc.start("arecord",
            ["-D", ARECORD_DEV, "-f", "S16_LE", "-r", "48000", "-c", "2", WAV_PATH])
        if not self._rec_proc.waitForStarted(3000):
            self._input.setPlaceholderText("麦克风启动失败")
            self._rec_proc = None
            return
        self._voice_state = 1
        self._btimer.start()
        self._style_voice()

    def _stop_and_recognize(self):
        self._btimer.stop()
        self._blink = False
        if self._rec_proc is not None:
            try:
                self._rec_proc.terminate()
                if not self._rec_proc.waitForFinished(3000):
                    self._rec_proc.kill()
                    self._rec_proc.waitForFinished(2000)
            except Exception:
                pass
            self._rec_proc = None
        self._voice_state = 2
        self._style_voice()
        self._rt = _RecognizeThread(WAV_PATH)
        self._rt.done.connect(self._on_voice_text)
        self._rt.fail.connect(self._on_voice_fail)
        self._rt.start()

    def _on_voice_text(self, text):
        self._voice_state = 0
        self._style_voice()
        if not text:
            self._input.setPlaceholderText("没听清，再说一次")
            return
        cur = self._input.text()
        self._input.setText((cur + text) if cur else text)
        self._input.setFocus()

    def _on_voice_fail(self, msg):
        self._voice_state = 0
        self._style_voice()
        self._input.setPlaceholderText("识别出错")

    # ---------- 展开 / 收起 ----------
    def popup(self):
        p = self.parent()
        if p is not None:
            bx, by = self._ball.x(), self._ball.y()
            x = min(max(0, bx - self.width() + self._ball.width()),
                    max(0, p.width() - self.width()))
            y = min(max(0, by - self.height() - 8),
                    max(0, p.height() - self.height()))
            self.move(x, y)
        self.show()
        self.raise_()
        self._input.setFocus()

    def collapse(self):
        if self._voice_state == 1:
            self._stop_and_recognize()
        self.hide()
        self._ball.show()
        self._ball.raise_()

    # ---------- 拖动(标题栏区域) ----------
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and ev.pos().y() <= 56:
            self._drag = True
            self._press = ev.globalPos()
            self._start = self.pos()
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag and self._press is not None:
            np = self._start + (ev.globalPos() - self._press)
            p = self.parent()
            if p is not None:
                x = max(0, min(np.x(), p.width() - self.width()))
                y = max(0, min(np.y(), p.height() - self.height()))
                self.move(x, y)
            else:
                self.move(np)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self._drag = False
        ev.accept()


# ============================================================
#  悬浮球
# ============================================================
class FloatingBall(QPushButton):
    """悬浮 AI 球：可拖动，点击展开对话面板"""
    def __init__(self, parent):
        super().__init__(parent)
        self._panel = None
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(64, 64)
        self.setText("💬")
        self.setStyleSheet(
            "QPushButton{background:#0a84ff;color:white;border:none;"
            "border-radius:32px;font-size:28px;}"
            "QPushButton:pressed{background:#0060df;}")
        # 拖动
        self._drag_active = False
        self._drag_moved = False
        self._press_pos = None
        self._btn_start = None
        self._DRAG_THRESH = 8
        self._user_moved = False

    def set_panel(self, panel):
        self._panel = panel

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_moved = False
            self._press_pos = ev.globalPos()
            self._btn_start = self.pos()
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag_active and self._press_pos is not None:
            delta = ev.globalPos() - self._press_pos
            if not self._drag_moved and delta.manhattanLength() >= self._DRAG_THRESH:
                self._drag_moved = True
            if self._drag_moved:
                np = self._btn_start + delta
                p = self.parent()
                if p is not None:
                    x = max(0, min(np.x(), p.width() - self.width()))
                    y = max(0, min(np.y(), p.height() - self.height()))
                    self.move(x, y)
                else:
                    self.move(np)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._drag_active:
            self._drag_active = False
            if self._drag_moved:
                self._user_moved = True
            else:
                self._on_click()
        ev.accept()

    def _on_click(self):
        if self._panel is not None:
            self.hide()
            self._panel.popup()

    def reposition(self):
        p = self.parent()
        if p is None:
            return
        if self._user_moved:
            x = max(0, min(self.x(), p.width() - self.width()))
            y = max(0, min(self.y(), p.height() - self.height()))
            self.move(x, y)
            self.raise_()
            return
        m = 24
        self.move(p.width() - self.width() - m,
                  p.height() - self.height() - m - 60)
        self.raise_()


def attach_voice_button(main_window):
    """在主窗口附加 AI 悬浮球 + 对话面板，返回球实例"""
    _load_title_font()
    ball = FloatingBall(main_window)
    panel = ChatPanel(main_window, ball)
    panel.hide()
    ball.set_panel(panel)
    ball.reposition()
    ball.show()
    ball.raise_()

    _orig_resize = main_window.resizeEvent
    def _resize(ev):
        if _orig_resize:
            _orig_resize(ev)
        ball.reposition()
    main_window.resizeEvent = _resize
    QTimer.singleShot(800, ball.reposition)
    QTimer.singleShot(2000, ball.reposition)
    return ball
