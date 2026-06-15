#!/usr/bin/env python3
"""
智能点锡与AOI检测系统 - RK3588版 (v2 全Scale自适应重构)

设计基准: 1024x600
Scale策略: 所有像素值 = base * scale, scale = screen_width / 1024
"""
import sys
import os
import cv2
import numpy as np
import time
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox, QTextEdit,
    QProgressBar, QLineEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont, QFontDatabase

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'material-640-640-v5n.rknn')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
sys.path.insert(0, os.path.join(BASE_DIR, 'vision'))

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Scale 工具
# ============================================================
def get_scale():
    """获取缩放因子，基准1024宽"""
    app = QApplication.instance()
    if app:
        screen = app.primaryScreen().geometry()
        return screen.width() / 1024.0
    return 1.0


def S(base_val):
    """缩放像素值"""
    return int(base_val * get_scale())


# ============================================================
# iOS风格 Stepper 控件
# ============================================================
class IOSStepper(QWidget):
    """iOS风格 [▼] value [▲] 控件，全scale自适应"""
    valueChanged = pyqtSignal(float)

    def __init__(self, min_val=0, max_val=100, value=50, step=1, decimals=0, parent=None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._value = value
        self._step = step
        self._decimals = decimals

        s = get_scale()
        btn_w, btn_h = int(32 * s), int(30 * s)
        edit_w, edit_h = int(50 * s), int(30 * s)
        font_sz = int(13 * s)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 左按钮 ▼
        self.btn_minus = QPushButton("\u25bc")
        self.btn_minus.setFixedSize(btn_w, btn_h)
        self.btn_minus.setStyleSheet(
            f"QPushButton {{ background: #e5e5ea; border: none; "
            f"border-top-left-radius: {int(8*s)}px; border-bottom-left-radius: {int(8*s)}px; "
            f"font-size: {font_sz}px; color: #007aff; font-family: Arial, sans-serif; }}"
            f"QPushButton:pressed {{ background: #c7c7cc; }}"
        )
        self.btn_minus.clicked.connect(self._dec)

        # 中间编辑框
        self.edit_value = QLineEdit()
        self.edit_value.setFixedSize(edit_w, edit_h)
        self.edit_value.setAlignment(Qt.AlignCenter)
        self.edit_value.setStyleSheet(
            f"QLineEdit {{ font-size: {font_sz}px; color: #1c1c1e; font-weight: 500; "
            f"border: none; border-top: 1px solid #d1d1d6; border-bottom: 1px solid #d1d1d6; "
            f"background: #ffffff; }}"
        )
        self.edit_value.editingFinished.connect(self._on_edit)

        # 右按钮 ▲
        self.btn_plus = QPushButton("\u25b2")
        self.btn_plus.setFixedSize(btn_w, btn_h)
        self.btn_plus.setStyleSheet(
            f"QPushButton {{ background: #e5e5ea; border: none; "
            f"border-top-right-radius: {int(8*s)}px; border-bottom-right-radius: {int(8*s)}px; "
            f"font-size: {font_sz}px; color: #007aff; font-family: Arial, sans-serif; }}"
            f"QPushButton:pressed {{ background: #c7c7cc; }}"
        )
        self.btn_plus.clicked.connect(self._inc)

        layout.addWidget(self.btn_minus)
        layout.addWidget(self.edit_value)
        layout.addWidget(self.btn_plus)

        self._update_display()

    def _update_display(self):
        if self._decimals == 0:
            self.edit_value.setText(str(int(self._value)))
        else:
            self.edit_value.setText(f"{self._value:.{self._decimals}f}")

    def _inc(self):
        self._value = min(self._max, self._value + self._step)
        self._update_display()
        self.valueChanged.emit(self._value)

    def _dec(self):
        self._value = max(self._min, self._value - self._step)
        self._update_display()
        self.valueChanged.emit(self._value)

    def _on_edit(self):
        try:
            v = float(self.edit_value.text())
            self._value = max(self._min, min(self._max, v))
        except ValueError:
            pass
        self._update_display()
        self.valueChanged.emit(self._value)

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = max(self._min, min(self._max, v))
        self._update_display()


# ============================================================
# 推理线程
# ============================================================
class InferenceThread(QThread):
    """YOLO推理线程"""
    result_ready = pyqtSignal(object, list, float)  # frame, detections, elapsed_ms

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.running = False
        self.cap = None
        self.rknn = None
        self.mode = 'camera'

    def init_model(self):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        self.rknn.load_rknn(self.model_path)
        self.rknn.init_runtime()

    def set_camera(self, cam_id):
        self.cap = cv2.VideoCapture(cam_id)
        self.mode = 'camera'

    def set_image(self, img):
        self._img = img
        self.mode = 'image'

    def run(self):
        from infer import letterbox, process_output, CONF_THRESH, NMS_THRESH, INPUT_SIZE
        self.running = True
        if self.rknn is None:
            self.init_model()

        while self.running:
            if self.mode == 'camera' and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
            elif self.mode == 'image':
                frame = self._img.copy()
            else:
                time.sleep(0.01)
                continue

            t0 = time.time()
            img_lb, r, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
            img_in = np.expand_dims(img_lb, axis=0)
            outputs = self.rknn.inference(inputs=[img_in])
            bboxes, scores, class_ids = process_output(outputs, frame.shape[:2], r, pad)
            elapsed = (time.time() - t0) * 1000

            detections = list(zip(bboxes, scores, class_ids)) if len(bboxes) > 0 else []
            self.result_ready.emit(frame, detections, elapsed)

            if self.mode == 'image':
                self.running = False
                break
            time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能点锡与AOI检测系统 - RK3588")

        # 全屏自适应
        screen = QApplication.primaryScreen().geometry()
        self.setFixedSize(screen.width(), screen.height())
        self.showFullScreen()

        self._scale = screen.width() / 1024.0

        # 状态
        self.current_mode = "solder"
        self.infer_thread = None
        self.current_frame = None
        self.current_detections = None
        self.path_result = None
        self._frozen = False

        self._build_ui()
        self._apply_style()

        # wmctrl强制全屏（兼容GNOME）
        QTimer.singleShot(500, self._force_fullscreen)

    def _force_fullscreen(self):
        try:
            subprocess.run(['wmctrl', '-r', ':ACTIVE:', '-b', 'add,fullscreen'], timeout=2,
                          capture_output=True)
        except Exception:
            pass

    # ----------------------------------------------------------
    # UI构建
    # ----------------------------------------------------------
    def _build_ui(self):
        s = self._scale
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(S(8), S(8), S(8), S(8))
        main_layout.setSpacing(S(8))

        # ===== 左侧：视频 + 状态栏 =====
        left_layout = QVBoxLayout()
        left_layout.setSpacing(S(6))
        left_layout.setContentsMargins(0, 0, 0, S(4))

        # 模式切换栏
        mode_bar = QHBoxLayout()
        self.btn_solder = QPushButton("点锡模式")
        self.btn_aoi = QPushButton("AOI检测")
        for btn in (self.btn_solder, self.btn_aoi):
            btn.setCheckable(True)
            btn.setFixedHeight(S(34))
        self.btn_solder.setChecked(True)
        mode_font_style = f"font-size: {S(13)}px; font-weight: 600;"
        self.btn_solder.setStyleSheet(f"QPushButton {{ {mode_font_style} }} QPushButton:checked {{ background: #007aff; color: white; border: none; {mode_font_style} }}")
        self.btn_aoi.setStyleSheet(f"QPushButton {{ {mode_font_style} }} QPushButton:checked {{ background: #007aff; color: white; border: none; {mode_font_style} }}")
        self.btn_solder.clicked.connect(lambda: self.switch_mode("solder"))
        self.btn_aoi.clicked.connect(lambda: self.switch_mode("aoi"))
        mode_bar.addWidget(self.btn_solder)
        mode_bar.addWidget(self.btn_aoi)
        left_layout.addLayout(mode_bar)

        # 视频显示区
        self.video_label = QLabel("点击 [开始] 启动摄像头")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(S(400), S(300))
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        left_layout.addWidget(self.video_label, 1)

        # 状态栏
        # 状态栏 - 固定高度Widget确保不被挤没
        status_widget = QWidget()
        status_widget.setFixedHeight(S(36))
        status_widget.setStyleSheet(f"background: white; border-radius: {S(6)}px {S(6)}px 0px 0px; border: none;")
        status_inner = QHBoxLayout(status_widget)
        status_inner.setContentsMargins(S(10), 0, S(10), 0)
        self.lbl_fps = QLabel("推理: -- ms")
        self.lbl_det = QLabel("检测: 0 个目标")
        self.lbl_path = QLabel("路径: -- 点")
        for lbl, color in [(self.lbl_fps, "#34c759"), (self.lbl_det, "#ff9500"), (self.lbl_path, "#007aff")]:
            lbl.setStyleSheet(f"color: {color}; font-weight: 600; font-size: {S(12)}px;")
        status_inner.addWidget(self.lbl_fps)
        status_inner.addWidget(self.lbl_det)
        status_inner.addWidget(self.lbl_path)
        left_layout.addWidget(status_widget)

        main_layout.addLayout(left_layout, 1)

        # ===== 右侧面板 =====
        right_widget = QWidget()
        right_widget.setFixedWidth(S(280))
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(S(4))

        # --- 控制按钮组 ---
        ctrl_group = QGroupBox("控制")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(S(4))

        btn_row1 = QHBoxLayout()
        self.btn_start = QPushButton("► 开始")
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setEnabled(False)
        btn_row1.addWidget(self.btn_start)
        btn_row1.addWidget(self.btn_stop)
        ctrl_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_capture = QPushButton("◎ 路径生成")
        self.btn_load = QPushButton("⊞ 加载图片")
        btn_row2.addWidget(self.btn_capture)
        btn_row2.addWidget(self.btn_load)
        ctrl_layout.addLayout(btn_row2)

        self.btn_execute = QPushButton("⚡ 执行点锡")
        self.btn_execute.setEnabled(False)
        ctrl_layout.addWidget(self.btn_execute)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(S(16))
        ctrl_layout.addWidget(self.progress_bar)

        right_layout.addWidget(ctrl_group)

        # --- 参数配置 ---
        param_group = QGroupBox("参数配置")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(S(6))

        def make_row(label_text, widget):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(S(80))
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(widget)
            return row

        self.spin_conf = IOSStepper(0.1, 0.9, 0.25, 0.05, 2)
        self.spin_spacing = IOSStepper(5, 50, 15, 5, 0)
        self.spin_dwell = IOSStepper(50, 1000, 200, 50, 0)
        self.spin_z = IOSStepper(1.0, 20.0, 5.0, 0.5, 1)

        param_layout.addLayout(make_row("置信度:", self.spin_conf))
        param_layout.addLayout(make_row("填充间距:", self.spin_spacing))
        param_layout.addLayout(make_row("停留(ms):", self.spin_dwell))
        param_layout.addLayout(make_row("安全高度:", self.spin_z))

        # 摄像头选择
        self.combo_cam = QComboBox()
        self.combo_cam.addItems(["21", "23", "25"])
        # 重写showPopup：弹出前扫描可用摄像头
        _orig_popup = self.combo_cam.showPopup
        def _custom_popup():
            if self._scan_cameras() is not False:
                _orig_popup()
        self.combo_cam.showPopup = _custom_popup
        self.combo_cam.setFixedSize(S(114), S(30))
        param_layout.addLayout(make_row("摄像头:", self.combo_cam))

        right_layout.addWidget(param_group)

        # --- 日志 ---
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, 1)

        main_layout.addWidget(right_widget)

        # --- 信号连接 ---
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_load.clicked.connect(self.load_image)
        self.btn_execute.clicked.connect(self.execute_solder)

    # ----------------------------------------------------------
    # 样式
    # ----------------------------------------------------------
    def _apply_style(self):
        s = self._scale
        fs_sm = S(10)
        fs_md = S(11)
        fs_lg = S(12)
        pad_sm = S(4)
        pad_md = S(8)
        rad = S(10)
        rad_sm = S(8)

        self.setStyleSheet(f"""
            QMainWindow {{ background: #f2f2f7; }}
            QGroupBox {{
                color: #1c1c1e; background: #ffffff;
                border-radius: {rad}px; border: 1px solid #e5e5ea;
                margin-top: {S(10)}px; padding: {pad_md}px {pad_md}px {S(6)}px {pad_md}px;
                font-weight: 600; font-size: {fs_md}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; subcontrol-position: top left; left: {pad_md}px; padding: 0 {pad_sm}px; top: {S(2)}px;
            }}
            QGroupBox QLabel {{ color: #1c1c1e; font-size: {fs_md}px; }}
            QPushButton {{
                background: #ffffff; color: #1c1c1e; border: 1px solid #d1d1d6;
                border-radius: {rad_sm}px; padding: {pad_sm}px {pad_md}px;
                font-size: {fs_md}px; font-weight: 500;
            }}
            QPushButton:pressed {{ background: #e5e5ea; }}
            QPushButton:checked {{ background: #007aff; color: white; border: none; }}
            QPushButton:disabled {{ background: #f2f2f7; color: #c7c7cc; border: 1px solid #e5e5ea; }}
            QComboBox {{
                background: #e5e5ea; color: #1c1c1e; border: none;
                border-radius: {rad_sm}px; padding-left: {S(46)}px;
                font-size: {fs_md}px;
            }}
            QComboBox::drop-down {{ width: 0; border: none; }}
            QComboBox QAbstractItemView {{
                background: white; border: 1px solid #d1d1d6;
                border-radius: {pad_sm}px; padding: {pad_sm}px;
                selection-background-color: #007aff; selection-color: white;
            }}
            QTextEdit {{
                background: #f9f9f9; border: 1px solid #e5e5ea;
                border-radius: {rad_sm}px; font-size: {fs_sm}px;
                padding: {pad_sm}px;
            }}
            QProgressBar {{
                border: 1px solid #d1d1d6; border-radius: {pad_sm}px;
                text-align: center; font-size: {fs_sm}px;
                background: #f2f2f7;
            }}
            QProgressBar::chunk {{ background: #34c759; border-radius: {S(3)}px; }}
        """)

        # 视频区样式
        self.video_label.setStyleSheet(
            f"background: #e5e5ea; color: #8e8e93; border-radius: {S(14)}px; font-size: {fs_lg}px;"
        )

        # 执行按钮特殊样式
        self.btn_execute.setStyleSheet(f"""
            QPushButton {{
                background: #007aff; color: white; border: none;
                border-radius: {rad_sm}px; padding: {S(6)}px;
                font-size: {fs_md}px; font-weight: 600;
            }}
            QPushButton:pressed {{ background: #0051d5; }}
            QPushButton:disabled {{ background: #a2c4f5; color: #e8e8e8; }}
        """)

    # ----------------------------------------------------------
    # 业务逻辑
    # ----------------------------------------------------------
    def log(self, msg):
        ts = time.strftime('%H:%M:%S')
        if msg.startswith("⚠") or msg.startswith("✗"):
            self.log_text.append(f'<span style="color:#ff3b30">[{ts}] {msg}</span>')
        else:
            self.log_text.append(f"[{ts}] {msg}")

    def switch_mode(self, mode):
        self.current_mode = mode
        self.btn_solder.setChecked(mode == "solder")
        self.btn_aoi.setChecked(mode == "aoi")
        if mode == "solder":
            self.btn_execute.setText("⚡ 执行点锡")
            self.log("⚙ 切换到 点锡模式")
        else:
            self.btn_execute.setText("🔍 执行AOI检测")
            self.log("⚙ 切换到 AOI检测模式")

    def start_camera(self):
        cam_id = int(self.combo_cam.currentText().replace("✓", "").strip())
        # 先验证摄像头能否打开
        import cv2 as _cv2
        _test = _cv2.VideoCapture(cam_id)
        if not _test.isOpened():
            self.log(f"⚠ 摄像头 {cam_id} 无法打开，请检查连接")
            return
        _test.release()
        # 验证通过，启动推理线程
        self.infer_thread = InferenceThread(MODEL_PATH)
        self.infer_thread.result_ready.connect(self.on_result)
        self.infer_thread.set_camera(cam_id)
        self.infer_thread.start()
        self._frozen = False
        self.path_result = None
        self.btn_execute.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_path.setText("路径: -- 点")
        self.log(f"✓ 摄像头 {cam_id} 已启动")

    def stop_camera(self):
        self._frozen = True

        if self.infer_thread and self.infer_thread.isRunning():
            self.infer_thread.running = False
            self.infer_thread.wait()
        if self.infer_thread and self.infer_thread.cap:
            self.infer_thread.cap.release()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def capture_frame(self):
        """路径生成: 用当前实时检测结果直接生成路径"""
        frame = self.current_frame.copy() if self.current_frame is not None else None
        detections = list(self.current_detections) if self.current_detections else []

        if frame is None:
            self.log("⚠ 无画面，请先启动摄像头")
            return

        if not detections:
            self.log("⚠ 当前无检测目标，无法生成路径")
            return

        # 冻结画面
        self._frozen = True
        self.stop_camera()

        try:
            from path_generator import generate_path, visualize_path
            bboxes = np.array([d[0] for d in detections])
            scores = np.array([d[1] for d in detections])
            class_ids = np.array([d[2] for d in detections])

            result = generate_path(bboxes, scores, class_ids,
                                   output_json=os.path.join(OUTPUT_DIR, 'path_output.json'),
                                   output_gcode=os.path.join(OUTPUT_DIR, 'path_output.gcode'))
            self.path_result = result

            vis_img = visualize_path(frame, result['points'],
                                     os.path.join(OUTPUT_DIR, 'path_visual.jpg'))
            self.display_frame(vis_img)
            self.lbl_path.setText(f"路径: {len(result['points'])} 点")
            self.log(f"✓ 路径生成完成: {len(result['points'])}点")
            self.btn_execute.setEnabled(True)
        except Exception as e:
            self.log(f"✗ 路径生成出错: {e}")
        finally:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def load_image(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.bmp)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.current_frame = img
                self.display_frame(img)
                self.log(f"⚙ 已加载: {os.path.basename(path)}")

    def execute_solder(self):
        if not self.path_result or not self.path_result.get('points'):
            self.log("⚠ 当前无可用路径，请先点击「路径生成」")
            return

        points = self.path_result['points']
        total = len(points)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self.btn_execute.setEnabled(False)
        self._exec_idx = 0
        self._exec_points = points

        self._exec_timer = QTimer()
        self._exec_timer.timeout.connect(self._exec_step)
        self._exec_timer.start(50)
        self.log(f"⚙ 开始执行点锡: {total} 点")

    def _exec_step(self):
        if self._exec_idx >= len(self._exec_points):
            self._exec_timer.stop()
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.btn_execute.setEnabled(True)
            self.log("✓ 点锡执行完成")
            return
        # TODO: 实际发送坐标到运动控制器
        self._exec_idx += 1
        self.progress_bar.setValue(self._exec_idx)

    def on_result(self, frame, detections, elapsed):
        if self._frozen:
            return
        self.current_frame = frame
        self.current_detections = detections
        self.lbl_fps.setText(f"推理: {elapsed:.1f} ms")
        self.lbl_det.setText(f"检测: {len(detections)} 个目标")

        # 画检测框
        vis = frame.copy()
        for bbox, score, cls_id in detections:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.display_frame(vis)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        label_size = self.video_label.size()
        pixmap = QPixmap.fromImage(qimg).scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)


    def _scan_cameras(self):
        """点击摄像头选择框时动态扫描设备，标记可用性"""
        # 运行中不允许切换
        if self.infer_thread and self.infer_thread.isRunning():
            self.log("⚠ 请先停止后再切换摄像头")
            return False
        import cv2
        current = self.combo_cam.currentText().strip()
        self.combo_cam.clear()
        candidates = [21, 23, 25]
        items = []
        for i in candidates:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                items.append(f"{i} ✓")
                cap.release()
            else:
                items.append(f"{i}")
        if not items:
            items = ["21"]
        self.combo_cam.addItems(items)
        # 恢复之前选中的
        for idx in range(self.combo_cam.count()):
            if current in self.combo_cam.itemText(idx):
                self.combo_cam.setCurrentIndex(idx)
                break

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    s = get_scale()
    app.setFont(QFont("PingFang SC", max(9, int(9 * s)), QFont.Medium))

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
