# -*- coding: utf-8 -*-
"""RK3588触摸屏 WiFi配置界面 - 经137直连线控制点锡机PC连WiFi"""
import sys, os, json, urllib.request
os.environ["QT_IM_MODULE"] = "qtvirtualkeyboard"
# 绕过板子代理直连PC
os.environ["no_proxy"] = "*"
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLineEdit, QLabel, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

AGENT = "http://192.168.137.222:8765"

def api_get(path, timeout=20):
    r = urllib.request.urlopen(AGENT + path, timeout=timeout)
    return json.loads(r.read().decode("utf-8"))

def api_post(path, obj, timeout=40):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(AGENT + path, data=data,
        headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(r.read().decode("utf-8"))

_LIVE_WORKERS = set()

def _track_worker(worker):
    _LIVE_WORKERS.add(worker)
    def release(w=worker):
        _LIVE_WORKERS.discard(w)
        w.deleteLater()
    worker.finished.connect(release)
    return worker

class Worker(QThread):
    done = pyqtSignal(object)
    err = pyqtSignal(str)
    def __init__(self, fn):
        super().__init__(); self.fn = fn
    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as e:
            self.err.emit(str(e))

class WifiConfig(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PC WiFi 配置")
        self.worker = None
        self._build()
        QTimer.singleShot(300, self.do_scan)
        QTimer.singleShot(500, self.refresh_status)

    def _build(self):
        self.setStyleSheet("""
            QWidget { background:#1e1e2e; color:#e0e0e0; font-size:22px; }
            QListWidget { background:#2a2a3a; border:1px solid #444; border-radius:8px; }
            QListWidget::item { padding:14px; }
            QListWidget::item:selected { background:#4a7aa8; color:#fff; }
            QLineEdit { background:#2a2a3a; border:2px solid #555; border-radius:8px; padding:12px; font-size:24px; }
            QPushButton { background:#3a6ea5; border:none; border-radius:8px; padding:14px 20px; font-size:22px; color:#fff; }
            QPushButton:pressed { background:#2d5580; }
            QLabel#status { font-size:20px; color:#8fd; }
            QLabel#title { font-size:28px; font-weight:bold; color:#fff; }
        """)
        root = QVBoxLayout(self); root.setContentsMargins(24,20,24,20); root.setSpacing(14)

        top = QHBoxLayout()
        title = QLabel("点锡机 PC 联网配置"); title.setObjectName("title")
        top.addWidget(title); top.addStretch()
        self.btn_scan = QPushButton("刷新列表"); self.btn_scan.clicked.connect(self.do_scan)
        top.addWidget(self.btn_scan)
        self.btn_close = QPushButton("关闭"); self.btn_close.clicked.connect(self.close)
        top.addWidget(self.btn_close)
        root.addLayout(top)

        self.status = QLabel("当前状态：查询中..."); self.status.setObjectName("status")
        root.addWidget(self.status)

        root.addWidget(QLabel("可用 WiFi（点击选择）："))
        self.listw = QListWidget()
        self.listw.itemClicked.connect(self.on_pick)
        root.addWidget(self.listw, 1)

        row = QHBoxLayout()
        self.ssid_edit = QLineEdit(); self.ssid_edit.setPlaceholderText("WiFi 名称")
        row.addWidget(self.ssid_edit, 1)
        root.addLayout(row)

        row2 = QHBoxLayout()
        self.pwd_edit = QLineEdit(); self.pwd_edit.setPlaceholderText("WiFi 密码")
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        row2.addWidget(self.pwd_edit, 1)
        self.btn_show = QPushButton("显示"); self.btn_show.setCheckable(True)
        self.btn_show.clicked.connect(self.toggle_pwd)
        row2.addWidget(self.btn_show)
        root.addLayout(row2)

        self.btn_conn = QPushButton("连接")
        self.btn_conn.clicked.connect(self.do_connect)
        root.addWidget(self.btn_conn)

    def toggle_pwd(self):
        self.pwd_edit.setEchoMode(QLineEdit.Normal if self.btn_show.isChecked() else QLineEdit.Password)

    def on_pick(self, item):
        self.ssid_edit.setText(item.data(Qt.UserRole))
        self.pwd_edit.setFocus()

    def _run(self, fn, done):
        self.worker = Worker(fn)
        _track_worker(self.worker)
        self.worker.done.connect(done)
        self.worker.err.connect(lambda m: self.status.setText("出错：" + m))
        self.worker.start()

    def do_scan(self):
        self.status.setText("正在扫描 WiFi...")
        self.btn_scan.setEnabled(False)
        def done(res):
            self.btn_scan.setEnabled(True)
            self.listw.clear()
            for n in res.get("networks", []):
                lock = "🔒" if n.get("auth","").lower() not in ("open","") else ""
                txt = "%s   %s  %s %s" % (n["ssid"], n.get("signal",""), n.get("band",""), lock)
                it = QListWidgetItem(txt)
                it.setData(Qt.UserRole, n["ssid"])
                self.listw.addItem(it)
            self.status.setText("扫描完成，共 %d 个" % self.listw.count())
        self._run(lambda: api_get("/scan"), done)

    def refresh_status(self):
        def done(res):
            if res.get("state","").lower().startswith("connect") or "已连接" in res.get("state",""):
                self.status.setText("已连接：%s   PC地址：%s" % (res.get("ssid",""), res.get("ip","")))
            else:
                self.status.setText("未连接")
        self._run(lambda: api_get("/status"), done)

    def do_connect(self):
        ssid = self.ssid_edit.text().strip()
        pwd = self.pwd_edit.text()
        if not ssid:
            QMessageBox.warning(self, "提示", "请先选择或输入 WiFi 名称"); return
        self.status.setText("正在连接 %s ..." % ssid)
        self.btn_conn.setEnabled(False)
        def done(res):
            self.btn_conn.setEnabled(True)
            if res.get("ok"):
                QMessageBox.information(self, "成功",
                    "已连接 %s\nPC 地址：%s\n\n手机浏览器访问：\nhttp://%s:8080"
                    % (res.get("ssid",""), res.get("ip",""), res.get("ip","")))
                self.status.setText("已连接：%s   PC地址：%s" % (res.get("ssid",""), res.get("ip","")))
            else:
                QMessageBox.warning(self, "失败",
                    "连接失败，请检查密码或信号\n状态：%s" % res.get("state",""))
                self.status.setText("连接失败")
        self._run(lambda: api_post("/connect", {"ssid": ssid, "pwd": pwd}), done)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = WifiConfig()
    geo = app.primaryScreen().geometry()
    w.setGeometry(geo)
    w.setWindowState(Qt.WindowFullScreen)
    w.show()
    w.raise_()
    w.activateWindow()
    QTimer.singleShot(200, w.showFullScreen)
    sys.exit(app.exec_())
