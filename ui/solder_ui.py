#!/usr/bin/env python3
"""
智能点锡与AOI检测系统 - RK3588版
==================================
硬件平台: 飞凌ELF2 (RK3588), HDMI触摸屏, USB摄像头, STM32执行系统
软件框架: PyQt5 + RKNNLite (NPU加速YOLOv5n)

功能模块:
    - 点锡模式: 实时检测焊盘 → 选中/编辑 → 路径规划 → G-code执行
    - AOI模式: 摄像头/图片输入 → 缺陷检测 → 红框标注显示
    - 执行系统: USB CDC心跳检测STM32在线状态

UI设计:
    - 设计基准: 1024x600, 全屏自适应缩放 (scale = screen_width / 1024)
    - 左侧: 视频/图像显示区(支持双指缩放) + 状态栏
    - 右侧: 模式切换 + 控制按钮 + 参数面板 + 日志区

采集流程 (点锡模式):
    1920x1080 MJPG采集 → 中心裁剪1080x1080 → 缩放640x640送NPU →
    检测坐标映射回原图 → 显示中心1640x1080区域(带标注)

作者: 梁泽欣
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
from PyQt5.QtGui import QPainter, QPainterPath, QImage, QPixmap, QFont, QFontDatabase

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINNING_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tinning_640_640.rknn')
AOI_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'aoi_640_640.rknn')
# 兼容旧变量名：点锡模型
MODEL_PATH = TINNING_MODEL_PATH
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
AOI_IMAGE_DIR = '/run/media/mmcblk1p1/AOI_Picture'
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



class PinchZoomLabel(QLabel):
    """支持双指缩放和拖动的图像Label（使用QPinchGesture）"""
    def __init__(self, *args, **kwargs):
        """初始化PinchZoomLabel：启用手势识别，设置缩放/平移参数"""
        super().__init__(*args, **kwargs)
        self.grabGesture(Qt.PinchGesture)
        self._current_display = None
        self._zoom = 1.0
        self._min_zoom = 1.0
        self._max_zoom = 5.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._original_pixmap = None
        self._gesture_in_progress = False  # 手势进行中用快速渲染
        self._gesture_active = False
        self._press_pos = None


    def _setRoundedPixmap(self, pixmap):
        """设置圆角裁剪的pixmap"""
        from PyQt5.QtGui import QPainter, QPainterPath, QPixmap
        from PyQt5.QtCore import QRectF
        radius = 14 * (self.width() / 1024) if self.width() > 0 else 14
        rounded = QPixmap(pixmap.size())
        rounded.fill(Qt.transparent)
        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(rounded.rect()), radius, radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        self._setRoundedPixmap(rounded)

    def setDisplayPixmap(self, pixmap):
        """供外部调用：存储原始pixmap并应用当前缩放"""
        self._original_pixmap = pixmap
        self._apply_transform()

    def reset_zoom(self):
        """重置缩放和平移到初始状态(1x, 无偏移)"""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        if self._original_pixmap:
            self._apply_transform()


    def paintEvent(self, event):
        """重写paintEvent: 使用QPainterPath实现圆角裁剪显示图像"""
        if self._current_display:
            from PyQt5.QtGui import QPainter, QPainterPath
            from PyQt5.QtCore import QRectF
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            radius = 14.0 * self.width() / 1024.0
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), radius, radius)
            painter.setClipPath(path)
            # 居中绘制pixmap
            pm = self._current_display
            x = (self.width() - pm.width()) // 2
            y = (self.height() - pm.height()) // 2
            painter.drawPixmap(x, y, pm)
            painter.end()
        else:
            super().paintEvent(event)

    def _apply_transform(self):
        """根据当前zoom/pan计算裁剪区域，生成_current_display并触发重绘"""
        if not self._original_pixmap:
            return
        pm = self._original_pixmap
        label_w = self.width() or pm.width()
        label_h = self.height() or pm.height()

        if self._zoom <= 1.0:
            # 正常显示，fitInView
            _mode = Qt.FastTransformation if self._gesture_in_progress else Qt.SmoothTransformation
            scaled = pm.scaled(label_w, label_h, Qt.KeepAspectRatio, _mode)
            self._current_display = scaled
            self.update()
            self._pan_x = 0.0
            self._pan_y = 0.0
            return

        # 放大：先缩放到zoom倍
        w = int(pm.width() * self._zoom)
        h = int(pm.height() * self._zoom)
        # 先fit到label再zoom
        base_scale = min(label_w / pm.width(), label_h / pm.height())
        w = int(pm.width() * base_scale * self._zoom)
        h = int(pm.height() * base_scale * self._zoom)
        _mode = Qt.FastTransformation if self._gesture_in_progress else Qt.SmoothTransformation
        scaled = pm.scaled(w, h, Qt.KeepAspectRatio, _mode)

        # 裁剪到label尺寸
        cx = w / 2 - self._pan_x
        cy = h / 2 - self._pan_y
        x = int(max(0, min(cx - label_w / 2, w - label_w)))
        y = int(max(0, min(cy - label_h / 2, h - label_h)))
        crop_w = min(label_w, w)
        crop_h = min(label_h, h)
        cropped = scaled.copy(x, y, crop_w, crop_h)
        self._current_display = cropped
        self.update()

    def mousePressEvent(self, ev):
        """记录按下位置，松手时判定是否为有效点击"""
        self._press_pos = ev.pos()
        self._gesture_active = False
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        """松手时：非手势中且移动距离小才算有效点击"""
        if (self._press_pos is not None
            and not self._gesture_active
            and self._original_pixmap and self.width() > 0):
            # 检查移动距离
            dx = ev.x() - self._press_pos.x()
            dy = ev.y() - self._press_pos.y()
            if (dx * dx + dy * dy) < 225:  # 15px以内算点击
                pm = self._original_pixmap
                label_w, label_h = self.width(), self.height()
                base_scale = min(label_w / pm.width(), label_h / pm.height())

                if self._zoom <= 1.0:
                    # 未放大：_current_display = fitInView scaled
                    disp = self._current_display
                    if disp:
                        off_x = (label_w - disp.width()) // 2
                        off_y = (label_h - disp.height()) // 2
                        px = ev.x() - off_x
                        py = ev.y() - off_y
                        if 0 <= px < disp.width() and 0 <= py < disp.height():
                            x_ratio = px / disp.width()
                            y_ratio = py / disp.height()
                        else:
                            x_ratio = y_ratio = -1
                    else:
                        x_ratio = y_ratio = -1
                else:
                    # 放大：需要反算到原图坐标
                    # scaled尺寸
                    sw = int(pm.width() * base_scale * self._zoom)
                    sh = int(pm.height() * base_scale * self._zoom)
                    # 裁剪起点(跟_apply_transform一致)
                    cx = sw / 2 - self._pan_x
                    cy = sh / 2 - self._pan_y
                    crop_x = int(max(0, min(cx - label_w / 2, sw - label_w)))
                    crop_y = int(max(0, min(cy - label_h / 2, sh - label_h)))
                    # _current_display居中绘制的偏移
                    disp = self._current_display
                    if disp:
                        off_x = (label_w - disp.width()) // 2
                        off_y = (label_h - disp.height()) // 2
                    else:
                        off_x = off_y = 0
                    # 点击在scaled pixmap上的坐标
                    sx = ev.x() - off_x + crop_x
                    sy = ev.y() - off_y + crop_y
                    # 映射到原图比例
                    x_ratio = sx / sw if sw > 0 else 0
                    y_ratio = sy / sh if sh > 0 else 0

                if 0 <= x_ratio <= 1 and 0 <= y_ratio <= 1:
                    main_win = self.window()
                    if hasattr(main_win, '_on_image_clicked'):
                        main_win._on_image_clicked(x_ratio, y_ratio)
        self._press_pos = None
        super().mouseReleaseEvent(ev)

    def event(self, ev):
        """事件分发：拦截手势事件交给_gesture_event处理"""
        if ev.type() == ev.Gesture:
            return self._gesture_event(ev)
        return super().event(ev)

    def _gesture_event(self, ev):
        """处理QPinchGesture：更新缩放比例和平移偏移"""
        self._gesture_active = True
        self._gesture_in_progress = True
        pinch = ev.gesture(Qt.PinchGesture)
        if pinch:
            if pinch.state() == Qt.GestureUpdated:
                scale_factor = pinch.scaleFactor()
                self._zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * scale_factor))
                # 平移
                delta = pinch.centerPoint() - pinch.lastCenterPoint()
                self._pan_x += delta.x()
                self._pan_y += delta.y()
                self._apply_transform()
            elif pinch.state() == Qt.GestureFinished:
                if self._zoom < 1.05:
                    self._zoom = 1.0
                    self._pan_x = 0.0
                    self._pan_y = 0.0
                    self._apply_transform()
            ev.accept()
            return True
        return False


class IOSStepper(QWidget):
    """iOS风格 [▼] value [▲] 控件，全scale自适应"""
    valueChanged = pyqtSignal(float)

    def __init__(self, min_val=0, max_val=100, value=50, step=1, decimals=0, parent=None):
        """初始化iOS风格步进器控件：[▼] 数值 [▲] 布局"""
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
        """更新中间数值显示文本"""
        if self._decimals == 0:
            self.edit_value.setText(str(int(self._value)))
        else:
            self.edit_value.setText(f"{self._value:.{self._decimals}f}")

    def _inc(self):
        """步进器加一步"""
        self._value = min(self._max, self._value + self._step)
        self._update_display()
        self.valueChanged.emit(self._value)

    def _dec(self):
        """步进器减一步"""
        self._value = max(self._min, self._value - self._step)
        self._update_display()
        self.valueChanged.emit(self._value)

    def _on_edit(self):
        """手动编辑数值后的回调：解析输入并约束范围"""
        try:
            v = float(self.edit_value.text())
            self._value = max(self._min, min(self._max, v))
        except ValueError:
            pass
        self._update_display()
        self.valueChanged.emit(self._value)

    def value(self):
        """获取当前步进器值"""
        return self._value

    def setValue(self, v):
        """设置步进器值(自动约束范围)"""
        self._value = max(self._min, min(self._max, v))
        self._update_display()


# ============================================================
# 推理线程
# ============================================================
class InferenceThread(QThread):
    """YOLO推理线程"""
    result_ready = pyqtSignal(object, list, float)  # frame, detections, elapsed_ms

    def __init__(self, model_path):
        """初始化推理线程：指定RKNN模型路径"""
        super().__init__()
        self.model_path = model_path
        self.running = False
        self.cap = None
        self.rknn = None
        self.mode = 'camera'

    def init_model(self):
        """加载RKNN模型到NPU"""
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        self.rknn.load_rknn(self.model_path)
        self.rknn.init_runtime()

    def set_camera(self, cam_id):
        """设置摄像头输入源：打开设备并配置1920x1080 MJPG 30fps"""
        self.cap = cv2.VideoCapture(cam_id)
        # 设置MJPG编码 + 1920x1080全高清采集
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.mode = 'camera'

    def set_image(self, img):
        """设置单张图片输入源(用于AOI加载图片模式)"""
        self._img = img
        self.mode = 'image'

    def run(self):
        """推理线程主循环：读帧→裁剪中心1080x1080→letterbox→NPU推理→坐标映射→emit结果"""
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

            h, w = frame.shape[:2]
            # 裁剪中心1080x1080用于模型推理
            crop_size = min(h, w)  # 1080 for 1920x1080
            cx, cy = w // 2, h // 2
            x1_crop = cx - crop_size // 2
            y1_crop = cy - crop_size // 2
            infer_crop = frame[y1_crop:y1_crop+crop_size, x1_crop:x1_crop+crop_size]

            t0 = time.time()
            img_lb, r, pad = letterbox(infer_crop, (INPUT_SIZE, INPUT_SIZE))
            img_in = np.expand_dims(img_lb, axis=0)
            outputs = self.rknn.inference(inputs=[img_in])
            bboxes, scores, class_ids = process_output(outputs, infer_crop.shape[:2], r, pad)
            elapsed = (time.time() - t0) * 1000

            # 检测坐标从crop坐标系映射回原图坐标系
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] += x1_crop
                bboxes[:, [1, 3]] += y1_crop

            # 裁剪中心1640x1080用于显示(保留更多上下文)
            disp_w = min(1640, w)
            x1_disp = cx - disp_w // 2
            display_frame = frame[0:h, x1_disp:x1_disp+disp_w].copy()

            # 检测坐标映射到display_frame坐标系
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] -= x1_disp

            detections = list(zip(bboxes, scores, class_ids)) if len(bboxes) > 0 else []
            self.result_ready.emit(display_frame, detections, elapsed)

            if self.mode == 'image':
                self.running = False
                break
            time.sleep(0.01)

    def stop(self):
        """停止推理线程并释放摄像头资源"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    """主窗口：智能点锡与AOI检测系统的核心UI，管理所有交互逻辑和子模块"""
    def __init__(self):
        """初始化主窗口：创建UI/信号连接/心跳定时器/状态变量"""
        super().__init__()
        self.setWindowTitle("智能点锡与AOI检测系统 - RK3588")

        # 全屏自适应
        screen = QApplication.primaryScreen().geometry()
        self.setFixedSize(screen.width(), screen.height())
        self.showFullScreen()

        # 运动系统心跳检测
        self._motor_online = False
        self._heartbeat_timer = QTimer(self)
        self._heartbeat_timer.timeout.connect(self._heartbeat_check)
        self._heartbeat_timer.start(10000)  # 60s
        # 启动时立刻检测一次
        QTimer.singleShot(3000, self._heartbeat_check)

        self._scale = screen.width() / 1024.0

        # 状态
        self.current_mode = "solder"
        self.infer_thread = None
        self.current_frame = None
        self.current_detections = None
        self.path_result = None
        self.loaded_aoi_image = None
        self.loaded_aoi_path = None
        self._frozen = False

        self._build_ui()
        self._apply_style()
        self._update_mode_controls()

        # wmctrl强制全屏（兼容GNOME）
        QTimer.singleShot(500, self._force_fullscreen)

    def _force_fullscreen(self):
        """延迟强制全屏(部分窗管首次不响应showFullScreen)"""
        try:
            subprocess.run(['wmctrl', '-r', ':ACTIVE:', '-b', 'add,fullscreen'], timeout=2,
                          capture_output=True)
        except Exception:
            pass

    # ----------------------------------------------------------
    # UI构建
    # ----------------------------------------------------------
    def _build_ui(self):
        """构建完整UI布局：左侧视频+状态栏，右侧控制面板+参数+日志"""
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
        self.video_label = PinchZoomLabel("点击 [开始] 启动摄像头")
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
        self.lbl_motor = QLabel("执行系统: --")
        self.lbl_motor.setStyleSheet(f"color: #8e8e93; font-weight: 600;")
        status_inner.addWidget(self.lbl_motor)
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
        self.btn_start.setFixedHeight(S(42))
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setFixedHeight(S(42))
        self.btn_stop.setEnabled(False)
        btn_row1.addWidget(self.btn_start)
        btn_row1.addWidget(self.btn_stop)
        ctrl_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_capture = QPushButton("◎ 路径生成")
        self.btn_capture.setFixedHeight(S(42))
        self.btn_load = QPushButton("⊞ 加载图片")
        self.btn_load.setFixedHeight(S(42))
        btn_row2.addWidget(self.btn_capture)
        btn_row2.addWidget(self.btn_load)
        ctrl_layout.addLayout(btn_row2)

        self.btn_execute = QPushButton("⚡ 执行点锡")
        self.btn_execute.setFixedHeight(S(42))
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
            """创建参数行布局：左侧标签+右侧控件"""
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
            """自定义摄像头选择下拉框弹出前的扫描回调"""
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
        self.log_text.setTextInteractionFlags(Qt.NoTextInteraction)
        self.log_text.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
        from PyQt5.QtWidgets import QScroller
        QScroller.grabGesture(self.log_text.viewport(), QScroller.TouchGesture)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, 1)

        main_layout.addWidget(right_widget)

        # --- 信号连接 ---
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_load.clicked.connect(self.load_image)
        self.btn_execute.clicked.connect(self.execute_action)

    # ----------------------------------------------------------
    # 样式
    # ----------------------------------------------------------
    def _apply_style(self):
        """应用iOS风格全局样式表：按钮/标签/进度条/日志区域"""
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
                font-size: {fs_lg}px; font-weight: 500;
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
                font-size: {fs_lg}px; font-weight: 600;
            }}
            QPushButton:pressed {{ background: #0051d5; }}
            QPushButton:disabled {{ background: #a2c4f5; color: #e8e8e8; }}
        """)

    # ----------------------------------------------------------
    # 业务逻辑
    # ----------------------------------------------------------
    def log(self, msg):
        """向日志区域追加一条带时间戳的消息(⚠开头显示红色)"""
        ts = time.strftime('%H:%M:%S')
        if msg.startswith("⚠") or msg.startswith("✗"):
            self.log_text.append(f'<span style="color:#ff3b30">[{ts}] {msg}</span>')
        else:
            self.log_text.append(f"[{ts}] {msg}")

    def _heartbeat_check(self):
        """每10s调用独立脚本检测执行系统是否在线"""
        import subprocess
        try:
            result = subprocess.run(
                ['python3', '/home/elf/solder_system/heartbeat_check.py'],
                timeout=3, capture_output=True
            )
            if result.returncode == 0:
                if not self._motor_online:
                    self.log("✓ 执行系统已上线")
                self._motor_online = True
                self.lbl_motor.setText("执行系统: 在线")
                self.lbl_motor.setStyleSheet("color: #34c759; font-weight: 600;")
            else:
                self._set_motor_offline()
        except Exception:
            self._set_motor_offline()

    def _set_motor_offline(self):
        """设置执行系统为离线状态"""
        if self._motor_online:
            self.log("⚠ 执行系统离线！")
        self._motor_online = False
        self.lbl_motor.setText("执行系统: 离线")
        self.lbl_motor.setStyleSheet("color: #ff3b30; font-weight: 600;")


    def switch_mode(self, mode):
        """切换工作模式：点锡(solder)↔AOI，停止当前推理并更新UI状态"""
        if mode == self.current_mode:
            return
        # 两种模式使用不同RKNN模型，切换时停止实时线程，避免模型错用
        if self.infer_thread and self.infer_thread.isRunning():
            self.stop_camera()
        self.current_mode = mode
        self.btn_solder.setChecked(mode == "solder")
        self.btn_aoi.setChecked(mode == "aoi")
        self.current_detections = None
        self.path_result = None
        self._update_mode_controls()
        if mode == "solder":
            self.log("⚙ 切换到 点锡模式")
        else:
            self.log("⚙ 切换到 AOI检测模式")
            self._ensure_aoi_dir()

    def _update_mode_controls(self):
        """根据当前模式更新按钮可用性和文字(点锡/AOI差异化)"""
        is_solder = self.current_mode == "solder"
        self.btn_capture.setText("◎ 路径生成" if is_solder else "◎ 锁定当前帧")
        self.btn_capture.setEnabled(True)
        self.btn_load.setEnabled(not is_solder)
        self.btn_execute.setText("⚡ 执行点锡" if is_solder else "🔍 执行AOI检测")
        self.btn_execute.setEnabled(False)
        if is_solder:
            self.lbl_path.setText("路径: -- 点")
        else:
            self.lbl_path.setText("AOI: --")

    def _ensure_aoi_dir(self):
        """确保AOI图片存储目录可用：优先SD卡，fallback到本地"""
        candidates = [
            AOI_IMAGE_DIR,
            os.path.join(BASE_DIR, 'AOI_Picture'),
        ]
        for path in candidates:
            try:
                os.makedirs(path, exist_ok=True)
                if os.access(path, os.R_OK | os.W_OK):
                    return path
            except Exception:
                continue
        return BASE_DIR

    def _lock_current_frame(self):
        """AOI模式：锁定当前检测后画面（带标注），关闭摄像头，复位按钮"""
        if self.current_frame is None:
            self.log("⚠ 无画面可锁定，请先启动摄像头")
            return
        # 用当前帧+检测结果生成带标注的画面
        frame = self.current_frame.copy()
        if self.current_detections:
            frame = self._draw_detections(frame, self.current_detections, color=(0, 0, 255), prefix="NG")
        self.loaded_aoi_image = frame
        self.loaded_aoi_path = "锁定帧"
        # 停止摄像头
        if self.infer_thread and self.infer_thread.isRunning():
            self.infer_thread.stop()
            self.infer_thread.wait()
            self.infer_thread = None
        # 复位开始/停止按钮
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # 冻结显示带标注的画面
        self._frozen = True
        self.display_frame(frame)
        n = len(self.current_detections) if self.current_detections else 0
        self.lbl_path.setText(f"AOI: 已锁定 ({n}个缺陷)")
        self.lbl_det.setText(f"缺陷: {n} 个")
        self.log(f"◎ 已锁定当前检测画面({n}个缺陷)，摄像头已关闭")

    def _current_model_path(self):
        """根据当前模式返回对应的RKNN模型路径(点锡/AOI)"""
        return TINNING_MODEL_PATH if self.current_mode == "solder" else AOI_MODEL_PATH

    def start_camera(self):
        """启动摄像头实时推理：验证设备→创建InferenceThread→开始"""
        cam_id = int(self.combo_cam.currentText().replace("✓", "").strip())
        # 先验证摄像头能否打开
        import cv2 as _cv2
        _test = _cv2.VideoCapture(cam_id)
        if not _test.isOpened():
            self.log(f"⚠ 摄像头 {cam_id} 无法打开，请检查连接")
            return
        _test.release()
        # 验证通过，启动推理线程
        self.infer_thread = InferenceThread(self._current_model_path())
        self.infer_thread.result_ready.connect(self.on_result)
        self.infer_thread.set_camera(cam_id)
        self.infer_thread.start()
        self._frozen = False
        self._edit_mode = False
        self._selection_mask = []
        self.path_result = None
        self.btn_execute.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_path.setText("路径: -- 点")
        self.log(f"✓ 摄像头 {cam_id} 已启动")

    def stop_camera(self):
        """停止摄像头：冻结画面，点锡模式进入编辑选中状态"""
        self._frozen = True

        if self.infer_thread and self.infer_thread.isRunning():
            self.infer_thread.running = False
            self.infer_thread.wait()
        if self.infer_thread and self.infer_thread.cap:
            self.infer_thread.cap.release()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        # 点锡模式停止后进入编辑选中模式
        if self.current_mode == 'solder' and self.current_detections:
            # 仅首次进入编辑模式时初始化为全选；已在编辑模式则保留之前的选中状态
            if not self._edit_mode or len(self._selection_mask) != len(self.current_detections):
                self._selection_mask = [True] * len(self.current_detections)
            self._edit_mode = True
            self._redraw_edit_frame()
            self.log("◎ 编辑模式：点击框可取消/恢复选中")


    def _redraw_edit_frame(self):
        """在编辑模式下重绘帧：选中的加蒙版，未选中的只有边框"""
        if self.current_frame is None:
            return
        vis = self.current_frame.copy()
        for i, det in enumerate(self.current_detections):
            bbox, score, cls_id = det
            x1, y1, x2, y2 = [int(v) for v in bbox]
            if self._selection_mask[i]:
                # 选中：画蒙版 + 边框
                overlay = vis.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 160, 0), -1)
                vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # 未选中：稍暗绿色细框，无蒙版
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        self.display_frame(vis)

    def _on_image_clicked(self, x_ratio, y_ratio):
        """图像区域被点击（坐标为相对于图像的0~1比例）"""
        if not self._edit_mode or not self.current_detections:
            return
        # 将比例坐标转为原图像素坐标
        h, w = self.current_frame.shape[:2]
        px, py = int(x_ratio * w), int(y_ratio * h)
        # 找到点击的是哪个框（从小到大，优先点小框）
        clicked_idx = -1
        min_area = float('inf')
        for i, det in enumerate(self.current_detections):
            bbox = det[0]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    min_area = area
                    clicked_idx = i
        if clicked_idx >= 0:
            self._selection_mask[clicked_idx] = not self._selection_mask[clicked_idx]
            state = "选中" if self._selection_mask[clicked_idx] else "取消"
            self.log(f"◎ 框{clicked_idx} {state}")
            self._redraw_edit_frame()

    def capture_frame(self):
        """点锡模式：生成路径；AOI模式：锁定当前帧"""
        if self.current_mode != "solder":
            self._lock_current_frame()
            return
        frame = self.current_frame.copy() if self.current_frame is not None else None
        detections = list(self.current_detections) if self.current_detections else []
        # 编辑模式下只用选中的检测结果
        if self._edit_mode and self._selection_mask and detections:
            detections = [d for d, sel in zip(detections, self._selection_mask) if sel]

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
        """AOI模式加载图片：打开文件对话框，显示图片到视频区"""
        if self.current_mode != "aoi":
            self.log("⚠ 加载图片仅在AOI检测模式可用")
            return
        from PyQt5.QtWidgets import QFileDialog
        aoi_dir = self._ensure_aoi_dir()
        path, _ = QFileDialog.getOpenFileName(self, "选择AOI图片", aoi_dir, "Images (*.jpg *.png *.bmp)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.loaded_aoi_image = img
                self.loaded_aoi_path = path
                self.current_frame = img
                self.display_frame(img)
                self.lbl_path.setText("AOI: 已载入图片")
                self.btn_execute.setEnabled(True)
                self.log(f"⚙ 已加载AOI图片: {os.path.basename(path)}")
            else:
                self.log(f"✗ 图片读取失败: {path}")

    def execute_action(self):
        """执行按钮分发：根据当前模式调用execute_solder或execute_aoi"""
        if self.current_mode == "solder":
            self.execute_solder()
        else:
            self.execute_aoi()

    def _infer_once(self, frame, model_path):
        """对单帧执行一次RKNN推理(阻塞)，返回(bboxes, scores, class_ids, elapsed_ms)"""
        from rknnlite.api import RKNNLite
        from infer import letterbox, process_output, INPUT_SIZE
        rknn = RKNNLite()
        rknn.load_rknn(model_path)
        rknn.init_runtime()
        t0 = time.time()
        img_lb, r, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
        img_in = np.expand_dims(img_lb, axis=0)
        outputs = rknn.inference(inputs=[img_in])
        bboxes, scores, class_ids = process_output(outputs, frame.shape[:2], r, pad)
        try:
            rknn.release()
        except Exception:
            pass
        elapsed = (time.time() - t0) * 1000
        return list(zip(bboxes, scores, class_ids)) if len(bboxes) > 0 else [], elapsed

    def _draw_detections(self, frame, detections, color=(0, 0, 255), prefix="DEFECT"):
        """在帧上绘制检测框和标签，返回标注后的图像副本"""
        vis = frame.copy()
        for bbox, score, cls_id in detections:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{prefix}{int(cls_id)} {score:.2f}"
            cv2.putText(vis, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis

    def execute_aoi(self):
        """执行AOI检测：用当前帧或锁定帧推理，结果标注红框显示"""
        frame = None
        source = ""
        if self.loaded_aoi_image is not None:
            frame = self.loaded_aoi_image.copy()
            source = os.path.basename(self.loaded_aoi_path or "导入图片")
        elif self.current_frame is not None:
            frame = self.current_frame.copy()
            source = "当前摄像头最后一帧"
        if frame is None:
            self.log("⚠ 无AOI输入：请先启动摄像头或加载图片")
            return
        # 冻结当前显示，避免实时线程覆盖AOI结果
        self._frozen = True
        if self.infer_thread and self.infer_thread.isRunning():
            self.stop_camera()
        try:
            detections, elapsed = self._infer_once(frame, AOI_MODEL_PATH)
            vis = self._draw_detections(frame, detections, color=(0, 0, 255), prefix="NG")
            out_path = os.path.join(OUTPUT_DIR, 'aoi_result.jpg')
            cv2.imwrite(out_path, vis)
            self.current_frame = frame
            self.current_detections = detections
            self.display_frame(vis)
            self.lbl_fps.setText(f"AOI: {elapsed:.1f} ms")
            self.lbl_det.setText(f"缺陷: {len(detections)} 个")
            self.lbl_path.setText("AOI: 完成")
            self.log(f"✓ AOI检测完成({source}): {len(detections)}个缺陷")
        except Exception as e:
            self.log(f"✗ AOI检测出错: {e}")
        finally:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_execute.setEnabled(True)

    def execute_solder(self):
        """执行点锡动作：将G-code通过串口发送到STM32(TODO)"""
        if self.current_mode != "solder":
            self.execute_aoi()
            return
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
        """点锡执行进度回调：更新进度条，逐步发送G-code指令"""
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
        """推理结果回调(InferenceThread信号)：更新状态栏+绘制检测框+显示帧"""
        if self._frozen:
            return
        self.current_frame = frame
        self.current_detections = detections
        self.lbl_fps.setText(f"推理: {elapsed:.1f} ms")
        self.lbl_det.setText(f"检测: {len(detections)} 个目标")

        # 画检测框：点锡绿色焊盘，AOI红色缺陷
        if self.current_mode == "aoi":
            vis = self._draw_detections(frame, detections, color=(0, 0, 255), prefix="NG")
            self.lbl_det.setText(f"缺陷: {len(detections)} 个")
        else:
            vis = self._draw_detections(frame, detections, color=(0, 255, 0), prefix="PAD")

        self.display_frame(vis)

    def display_frame(self, frame):
        """将OpenCV BGR帧转为QPixmap并显示到PinchZoomLabel(保持缩放状态)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        label_size = self.video_label.size()
        pixmap = QPixmap.fromImage(qimg).scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setDisplayPixmap(pixmap)


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
        """窗口关闭事件：停止推理线程，释放资源"""
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
