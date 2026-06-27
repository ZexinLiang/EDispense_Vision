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
    QDialog,
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
TINNING_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pad.rknn')
AOI_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'qs.rknn')
# 兼容旧变量名：点锡模型
MODEL_PATH = TINNING_MODEL_PATH
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
XY_CALIB_PATH = os.path.join(CONFIG_DIR, 'xy_calib.json')
AOI_IMAGE_DIR = '/media/elf/OPI_BOOT/AOI_Picture'
SOLDER_Z_DOWN_STEPS = 890
SOLDER_Z_LIFT_POS = 600
SOLDER_SQUEEZE_COUNT = 1
SOLDER_STEP_TIMEOUT_MS = 12000
SOLDER_HOME_TIMEOUT_MS = 20000
SOLDER_TIMER_INTERVAL_MS = 50
SOLDER_XY_MIN_WAIT_MS = 1200
SOLDER_Z_DOWN_MIN_WAIT_MS = 0
SOLDER_SQUEEZE_MIN_WAIT_MS = 500
SOLDER_Z_UP_MIN_WAIT_MS = 1000
SOLDER_HOME_MIN_WAIT_MS = 3000
sys.path.insert(0, os.path.join(BASE_DIR, 'vision'))
sys.path.insert(0, BASE_DIR)  # motor_control.py在项目根目录

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 相机畸变矫正 (仅 video21 顶视相机)
# ============================================================
UNDISTORT_CAM_ID = 21
CALIB_NPZ_PATH = os.path.join(CONFIG_DIR, 'calibration_result_4.npz')


class _Undistorter:
    """video21 顶视相机畸变矫正器。
    懒加载标定参数(mtx/dist), 按分辨率预计算 remap 表, 对每帧整图做 remap。
    注意: XY标定与点锡像素->机床映射均应基于矫正后的图像, 故更换本标定后必须重新做XY标定。"""

    def __init__(self):
        self._maps = {}      # (w, h) -> (map1, map2)
        self._mtx = None
        self._dist = None
        self._loaded = False
        self._ok = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            data = np.load(CALIB_NPZ_PATH)
            self._mtx = data['mtx']
            self._dist = data['dist']
            self._ok = True
            try:
                err = float(data['reprojection_error'])
                print(f"[undistort] loaded {CALIB_NPZ_PATH}, reproj_err={err:.4f}")
            except Exception:
                print(f"[undistort] loaded {CALIB_NPZ_PATH}")
        except Exception as e:
            print(f"[undistort] load failed: {e} -> video21 不做畸变矫正")
            self._ok = False

    def apply(self, frame):
        """对整帧BGR做畸变矫正; 标定缺失/加载失败时原样返回。"""
        if frame is None:
            return frame
        self._ensure_loaded()
        if not self._ok:
            return frame
        h, w = frame.shape[:2]
        key = (w, h)
        maps = self._maps.get(key)
        if maps is None:
            # 用原相机矩阵作为新矩阵, 保持中心尺度不变(边缘可能出现少量黑边)
            map1, map2 = cv2.initUndistortRectifyMap(
                self._mtx, self._dist, None, self._mtx, (w, h), cv2.CV_16SC2)
            maps = (map1, map2)
            self._maps[key] = maps
        return cv2.remap(frame, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)


_undistorter = _Undistorter()


def undistort_cam21(frame, cam_id):
    """仅当 cam_id == 21 时对帧做畸变矫正, 其余相机原样返回。"""
    try:
        if int(cam_id) == UNDISTORT_CAM_ID:
            return _undistorter.apply(frame)
    except (ValueError, TypeError):
        pass
    return frame


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




class NumPadDialog(QDialog):
    """触摸屏数字键盘弹窗(iOS风格,深色)。返回输入的数值字符串。"""
    def __init__(self, parent=None, title="输入数值", init_value="", allow_negative=False):
        """初始化数字键盘弹窗：标题/初值/是否允许负号"""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self._value = str(init_value)
        self._allow_neg = allow_negative
        s = get_scale()
        self.setFixedSize(int(320*s), int(420*s))
        self.setStyleSheet(f"QDialog {{ background: #1c1c1e; border-radius: {int(16*s)}px; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(int(16*s),int(16*s),int(16*s),int(16*s))
        lay.setSpacing(int(10*s))

        # 标题
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet(f"color:#8e8e93; font-size:{int(13*s)}px; font-weight:600;")
        lay.addWidget(lbl_title)

        # 显示框
        self.display = QLabel(self._value or "0")
        self.display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.display.setStyleSheet(
            f"background:#000; color:#fff; font-size:{int(28*s)}px; font-weight:700;"
            f"border-radius:{int(10*s)}px; padding:{int(8*s)}px {int(14*s)}px;")
        self.display.setFixedHeight(int(56*s))
        lay.addWidget(self.display)

        # 按键网格
        grid = QGridLayout()
        grid.setSpacing(int(8*s))
        keys = [('1',0,0),('2',0,1),('3',0,2),
                ('4',1,0),('5',1,1),('6',1,2),
                ('7',2,0),('8',2,1),('9',2,2),
                ('.',3,0),('0',3,1),('⌫',3,2)]
        for txt,row,col in keys:
            b = QPushButton(txt)
            b.setFixedSize(int(88*s),int(58*s))
            is_special = txt in ('⌫','.')
            b.setStyleSheet(f"""
                QPushButton {{ background:{'#3a3a3c' if is_special else '#2c2c2e'}; color:#fff;
                    font-size:{int(22*s)}px; font-weight:600; border:none;
                    border-radius:{int(10*s)}px; }}
                QPushButton:pressed {{ background:#0a84ff; }}
            """)
            b.clicked.connect(lambda _,t=txt: self._on_key(t))
            grid.addWidget(b,row,col)
        lay.addLayout(grid)

        # 底部: 取消 + 确认
        btn_row = QHBoxLayout()
        btn_row.setSpacing(int(8*s))
        btn_cancel = QPushButton("取消")
        btn_cancel.setFixedHeight(int(50*s))
        btn_cancel.setStyleSheet(f"QPushButton {{ background:#3a3a3c; color:#fff; font-size:{int(16*s)}px; border:none; border-radius:{int(10*s)}px; }} QPushButton:pressed {{ background:#555; }}")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("确认")
        btn_ok.setFixedHeight(int(50*s))
        btn_ok.setStyleSheet(f"QPushButton {{ background:#0a84ff; color:#fff; font-size:{int(16*s)}px; font-weight:600; border:none; border-radius:{int(10*s)}px; }} QPushButton:pressed {{ background:#0060df; }}")
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        lay.addLayout(btn_row)

    def _on_key(self, t):
        """数字键盘按键回调：t为按键字符(0-9/./⌫/C)，更新输入缓冲"""
        if t == '⌫':
            self._value = self._value[:-1]
        elif t == '.':
            if '.' not in self._value:
                self._value = (self._value or "0") + '.'
        else:
            if self._value == "0":
                self._value = t
            else:
                self._value += t
        self.display.setText(self._value or "0")

    def get_value(self):
        """返回用户输入的数值字符串"""
        return self._value or "0"


class TouchScrollTextEdit(QTextEdit):
    """支持触摸拖动滚动的日志文本框(QScroller惯性滚动,丝滑)

    用QScroller.grabGesture实现触摸惯性滚动。
    注: 经验证GNOME OSK只弹一次是系统bug,与QScroller无关,故可放心使用。
    """
    def __init__(self, *args, **kwargs):
        """初始化: 启用双指缩放手势, 初始化缩放/平移状态"""
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        from PyQt5.QtWidgets import QScroller
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
        QScroller.grabGesture(self.viewport(), QScroller.LeftMouseButtonGesture)


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
        """供外部调用：存储原始pixmap并应用当前缩放(手势中跳过避免卡顿)"""
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
            _mode = Qt.SmoothTransformation
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
        _mode = Qt.SmoothTransformation
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
    remote_error = pyqtSignal(str)  # 远程推理单帧失败时发出错误信息

    def __init__(self, model_path):
        """初始化推理线程：指定RKNN模型路径"""
        super().__init__()
        self.model_path = model_path
        self.conf_thresh = 0.25  # 由UI的置信度spin动态更新
        self.running = False
        self.cap = None
        self.rknn = None
        self.mode = 'camera'
        self.cam_id = None
        self.use_remote = False       # True则走外部网口推理
        self.remote_client = None     # RemoteInferClient实例

    def init_model(self):
        """加载RKNN模型到NPU"""
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        self.rknn.load_rknn(self.model_path)
        self.rknn.init_runtime()

    def set_camera(self, cam_id):
        """设置摄像头输入源：打开设备并配置1920x1080 MJPG 30fps"""
        self.cam_id = int(cam_id)
        self.cap = cv2.VideoCapture(self.cam_id)
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
        from infer import infer
        self.running = True
        if self.rknn is None:
            self.init_model()

        while self.running:
            if self.mode == 'camera' and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                # video21顶视相机畸变矫正(其余相机原样)
                frame = undistort_cam21(frame, self.cam_id)
            elif self.mode == 'image':
                frame = self._img.copy()
            else:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            # === 单次推理：裁剪中心1080x1080正方形ROI → resize到1088 → NPU推理 ===
            t0 = time.time()
            crop_size = min(h, w)  # 1080 for 1920x1080
            cx, cy = w // 2, h // 2
            x1_crop = cx - crop_size // 2
            y1_crop = cy - crop_size // 2
            infer_crop = frame[y1_crop:y1_crop+crop_size, x1_crop:x1_crop+crop_size]

            try:
                if self.use_remote and self.remote_client is not None:
                    bboxes, scores, class_ids = self.remote_client.infer(
                        infer_crop, conf_thresh=self.conf_thresh)
                else:
                    bboxes, scores, class_ids = infer(
                        self.rknn, infer_crop, conf_thresh=self.conf_thresh)
            except Exception as e:
                # 远程单帧失败: 报警并跳过本帧, 保持外部模式等下一帧重试
                if self.use_remote:
                    self.remote_error.emit(str(e))
                    time.sleep(0.05)
                    continue
                raise
            elapsed = (time.time() - t0) * 1000
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] += x1_crop
                bboxes[:, [1, 3]] += y1_crop

            # 裁剪显示区域: 中心1640x1080
            disp_w = min(1640, w)
            x1_disp = cx - disp_w // 2
            self.disp_offset_x = x1_disp  # 记录显示裁剪x偏移, 供路径生成还原全图坐标
            display_frame = frame[0:h, x1_disp:x1_disp+disp_w].copy()
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
# 摄像头预览线程 (仅取流显示, 不做推理, 用于调试模式针尖相机)
# ============================================================
class CameraPreviewThread(QThread):
    """轻量摄像头预览线程: 只读帧并发信号, 不加载RKNN/不推理(省NPU)。
    用于调试模式下显示针尖校准相机画面。"""
    frame_ready = pyqtSignal(object)  # frame(BGR)
    resolution_ready = pyqtSignal(int, int)  # 实际分辨率 w, h
    open_failed = pyqtSignal(int)  # 打开失败, 携带cam_id

    def __init__(self, cam_id):
        super().__init__()
        self.cam_id = cam_id
        self.running = False
        self.cap = None

    def run(self):
        """打开摄像头(MJPG 1920x1080)循环取帧"""
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            self.open_failed.emit(self.cam_id)
            return
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # 回读实际生效分辨率
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution_ready.emit(aw, ah)
        self.running = True
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = undistort_cam21(frame, self.cam_id)
                    self.frame_ready.emit(frame)
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.05)
            time.sleep(0.02)

    def stop(self):
        """停止预览并释放摄像头"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


class HealthCheckThread(QThread):
    """后台推理系统健康探测线程(避免在GUI主线程做阻塞网络请求导致卡顿)"""
    status_changed = pyqtSignal(bool)  # 仅在在线状态变化时发出
    status_tick = pyqtSignal(bool)     # 每次探测结果(用于掉线时持续触发回退)

    def __init__(self, client, interval=1.0):
        super().__init__()
        self.client = client
        self.interval = interval
        self.running = False
        self._last = None

    def run(self):
        self.running = True
        while self.running:
            online = self.client.check_health()
            self.status_tick.emit(online)
            if online != self._last:
                self._last = online
                self.status_changed.emit(online)
            time.sleep(self.interval)

    def stop(self):
        self.running = False
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
        self._heartbeat_timer.start(2000)  # 60s
        # 启动时立刻检测一次
        QTimer.singleShot(3000, self._heartbeat_check)

        # 运动控制器(USB CDC) + 主线程状态轮询
        self._current_x = 0.0
        self._current_y = 0.0
        self._current_z = 0
        try:
            from motor_control import MotorController
            self._motor = MotorController(logger=lambda m: (self.log(m) if hasattr(self,'log_text') else print('[motor]',m)))
            self._motor.connect()
        except Exception as _e:
            self.log(f"✗ 运动控制器初始化失败: {_e}")
            self._motor = None
        self._motor_poll_timer = QTimer(self)
        self._motor_poll_timer.timeout.connect(self._poll_motor_state)
        self._motor_poll_timer.start(200)

        self._scale = screen.width() / 1024.0

        # 外部网口推理 (Win11推理服务)
        self._use_remote = False          # 当前是否使用外部推理
        self._remote_online = False       # 推理系统在线状态(health轮询结果)
        from remote_infer import RemoteInferClient
        self._remote_client = RemoteInferClient("http://192.168.137.222:8000")
        # health探测放后台线程(避免阻塞GUI主线程导致渲染/缩放卡顿)
        self._health_thread = HealthCheckThread(self._remote_client, interval=1.0)
        self._health_thread.status_changed.connect(self._on_remote_status_changed)
        self._health_thread.status_tick.connect(self._on_remote_status_tick)
        self._health_thread.start()

        # 状态
        self.current_mode = "solder"
        self.infer_thread = None
        self.current_frame = None
        self.current_detections = None
        self.path_result = None
        self.loaded_aoi_image = None
        self.loaded_aoi_path = None
        self._frozen = False
        self._tip_preview = None   # 针尖校准相机预览线程(调试模式)
        # XY标定状态
        self._xy_calib_active = False
        self._xy_calib_state = 'idle'   # idle/picking/locked/aligning
        self._xy_calib_frame = None     # 冻结的顶部相机帧(BGR)
        self._xy_calib_cur_px = None    # 当前候选像素点(u,v)
        self._xy_calib_pairs = []       # [(u,v,X,Y), ...] 已记录的标定对
        self._xy_calib_M = None         # 解算出的2x3仿射矩阵(像素→机床)
        self._load_xy_calib()
        # XY测试(验证标定准不准)状态
        self._xytest_state = 'idle'     # idle/picking
        self._xytest_frame = None       # 冻结顶图
        self._xytest_px = None          # 选中像素(u,v)
        self._xytest_xy = None          # 换算出的机床(X,Y)

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
        self.btn_debug = QPushButton("调试")
        for btn in (self.btn_solder, self.btn_aoi, self.btn_debug):
            btn.setCheckable(True)
            btn.setFixedHeight(S(34))
        self.btn_solder.setChecked(True)
        mode_font_style = f"font-size: {S(13)}px; font-weight: 600;"
        self.btn_solder.setStyleSheet(f"QPushButton {{ {mode_font_style} }} QPushButton:checked {{ background: #007aff; color: white; border: none; {mode_font_style} }}")
        self.btn_aoi.setStyleSheet(f"QPushButton {{ {mode_font_style} }} QPushButton:checked {{ background: #007aff; color: white; border: none; {mode_font_style} }}")
        self.btn_debug.setStyleSheet(f"QPushButton {{ {mode_font_style} }} QPushButton:checked {{ background: #007aff; color: white; border: none; {mode_font_style} }}")
        self.btn_solder.clicked.connect(lambda: self.switch_mode("solder"))
        self.btn_aoi.clicked.connect(lambda: self.switch_mode("aoi"))
        self.btn_debug.clicked.connect(lambda: self.switch_mode("debug"))
        mode_bar.addWidget(self.btn_solder)
        mode_bar.addWidget(self.btn_aoi)
        mode_bar.addWidget(self.btn_debug)
        left_layout.addLayout(mode_bar)

        # 针尖校准相机切换条(仅调试模式可见, 用于针头/板面标定预览)
        self._tip_cam_bar = QWidget()
        tip_cam_lay = QHBoxLayout(self._tip_cam_bar)
        tip_cam_lay.setContentsMargins(0, 0, 0, 0)
        tip_cam_lay.setSpacing(S(8))
        tip_lbl = QLabel("针尖校准相机:")
        tip_lbl.setStyleSheet(f"font-size: {S(12)}px; font-weight: 600;")
        self.combo_tip_cam = QComboBox()
        self.combo_tip_cam.addItems(["21", "23", "25"])
        self.combo_tip_cam.setCurrentText("23")   # 默认针尖相机 /dev/video23
        self.combo_tip_cam.setFixedHeight(S(30))
        self.combo_tip_cam.currentTextChanged.connect(self._on_tip_cam_changed)
        self.btn_tip_open = QPushButton("打开")
        self.btn_tip_open.setFixedHeight(S(30))
        self.btn_tip_open.setStyleSheet(f"font-size: {S(12)}px; background: #34c759; color: white; border: none; border-radius: {S(6)}px;")
        self.btn_tip_open.clicked.connect(self._open_tip_cam)
        self.btn_tip_close = QPushButton("关闭")
        self.btn_tip_close.setFixedHeight(S(30))
        self.btn_tip_close.setStyleSheet(f"font-size: {S(12)}px; background: #ff3b30; color: white; border: none; border-radius: {S(6)}px;")
        self.btn_tip_close.clicked.connect(self._close_tip_cam)
        self.btn_tip_close.setEnabled(False)
        tip_cam_lay.addWidget(tip_lbl)
        tip_cam_lay.addWidget(self.combo_tip_cam, 2)
        tip_cam_lay.addWidget(self.btn_tip_open, 2)
        tip_cam_lay.addWidget(self.btn_tip_close, 2)
        self._tip_cam_bar.setVisible(False)
        left_layout.addWidget(self._tip_cam_bar)

        # 视频显示区
        self.video_label = PinchZoomLabel("点击 [开始] 启动摄像头")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(S(400), S(300))
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        left_layout.addWidget(self.video_label, 1)

        # 标定按钮区(仅调试模式可见): XY标定 / Z补偿
        self._calib_bar = QWidget()
        calib_v = QVBoxLayout(self._calib_bar)
        calib_v.setContentsMargins(0, 0, 0, 0)
        calib_v.setSpacing(S(4))
        # 状态标签
        self._lbl_calib_status = QLabel("标定: 空闲")
        self._lbl_calib_status.setStyleSheet(f"font-size: {S(12)}px; font-weight: 600; color: #5856d6;")
        calib_v.addWidget(self._lbl_calib_status)
        # 主按钮行: XY标定 / Z补偿
        calib_main = QHBoxLayout()
        calib_main.setSpacing(S(6))
        self.btn_xy_calib = QPushButton("XY标定")
        self.btn_xy_calib.setFixedHeight(S(34))
        self.btn_xy_calib.setStyleSheet(f"font-size: {S(12)}px; background: #5856d6; color: white; border: none; border-radius: {S(6)}px;")
        self.btn_xy_calib.clicked.connect(self._xy_calib_start)
        self.btn_z_calib = QPushButton("Z补偿")
        self.btn_z_calib.setFixedHeight(S(34))
        self.btn_z_calib.setStyleSheet(f"font-size: {S(12)}px; background: #8e8e93; color: white; border: none; border-radius: {S(6)}px;")
        self.btn_z_calib.clicked.connect(lambda: self.log("⚠ Z补偿功能待实现"))
        calib_main.addWidget(self.btn_xy_calib)
        calib_main.addWidget(self.btn_z_calib)
        calib_v.addLayout(calib_main)
        # XY标定上下文按钮行(标定进行中才启用)
        calib_ctx = QHBoxLayout()
        calib_ctx.setSpacing(S(6))
        self.btn_calib_lock = QPushButton("锁定点")
        self.btn_calib_align = QPushButton("手动标定")
        self.btn_calib_record = QPushButton("确认记录")
        self.btn_calib_cancel = QPushButton("取消/重置")
        for b, col in ((self.btn_calib_lock, "#34c759"), (self.btn_calib_align, "#007aff"),
                       (self.btn_calib_record, "#ff9500"), (self.btn_calib_cancel, "#ff3b30")):
            b.setFixedHeight(S(32))
            b.setStyleSheet(f"font-size: {S(11)}px; background: {col}; color: white; border: none; border-radius: {S(6)}px;")
            b.setEnabled(False)
            calib_ctx.addWidget(b)
        self.btn_calib_lock.clicked.connect(self._xy_calib_lock)
        self.btn_calib_align.clicked.connect(self._xy_calib_align)
        self.btn_calib_record.clicked.connect(self._xy_calib_record)
        self.btn_calib_cancel.clicked.connect(self._xy_calib_cancel)
        calib_v.addLayout(calib_ctx)
        self._calib_bar.setVisible(False)
        left_layout.addWidget(self._calib_bar)

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
        self.lbl_infersys = QLabel("推理系统: 离线")
        self.lbl_infersys.setStyleSheet(f"color: #8e8e93; font-weight: 600;")
        status_inner.addWidget(self.lbl_infersys)
        left_layout.addWidget(status_widget)

        main_layout.addLayout(left_layout, 1)

        # ===== 右侧面板 =====
        self._right_widget = QWidget()
        right_widget = self._right_widget
        right_widget.setFixedWidth(S(280))
        self._right_layout = QVBoxLayout(right_widget)
        right_layout = self._right_layout
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
        # 点锡模式下btn_load复用为"暂停"; 暂停后分裂出"继续"+"终止"(默认隐藏)
        self.btn_resume = QPushButton("▶ 继续")
        self.btn_resume.setFixedHeight(S(42))
        self.btn_resume.setVisible(False)
        self.btn_terminate = QPushButton("■ 终止")
        self.btn_terminate.setFixedHeight(S(42))
        self.btn_terminate.setVisible(False)
        btn_row2.addWidget(self.btn_capture)
        btn_row2.addWidget(self.btn_load)
        btn_row2.addWidget(self.btn_resume)
        btn_row2.addWidget(self.btn_terminate)
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
        self.spin_conf.valueChanged.connect(self._on_conf_changed)
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
        self.combo_cam.setFixedSize(S(70), S(30))
        # 摄像头行: 左标签+压缩的combo + 右侧"推理切换"按钮(本地/外部, 共用全局状态)
        cam_row = QHBoxLayout()
        cam_lbl = QLabel("摄像头:")
        cam_lbl.setFixedWidth(S(80))
        cam_row.addWidget(cam_lbl)
        cam_row.addWidget(self.combo_cam)
        cam_row.addStretch()
        self.btn_infer_src = QPushButton("本地推理")
        self.btn_infer_src.setFixedSize(S(100), S(30))
        self.btn_infer_src.clicked.connect(self._toggle_infer_source)
        cam_row.addWidget(self.btn_infer_src)
        param_layout.addLayout(cam_row)

        right_layout.addWidget(param_group)

        # --- 日志 ---
        self.log_group = QGroupBox("日志")
        log_group = self.log_group
        log_layout = QVBoxLayout(log_group)
        self.log_text = TouchScrollTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, 1)

        main_layout.addWidget(right_widget)
        
        # 创建调试面板（隐藏在右侧面板同位置）
        self._debug_widget = self._create_debug_panel()
        main_layout.addWidget(self._debug_widget)
        self._debug_widget.setVisible(False)

        # --- 信号连接 ---
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_load.clicked.connect(self._on_btn_load_clicked)
        self.btn_resume.clicked.connect(self._on_solder_resume)
        self.btn_terminate.clicked.connect(self._on_solder_terminate)
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
        """向日志区追加带时间戳的消息。⚠/✗ 开头标红(警告/错误), 其余正常色。"""
        if not hasattr(self, 'log_text'):
            print('[log]', msg)
            return
        ts = time.strftime('%H:%M:%S')
        if msg.startswith("⚠") or msg.startswith("✗"):
            self.log_text.append(f'<span style="color:#ff3b30">[{ts}] {msg}</span>')
        else:
            self.log_text.append(f"[{ts}] {msg}")

    def _heartbeat_check(self):
        """每10s基于STM32上行帧判定执行系统在线状态(不再依赖回显)。"""
        online = bool(self._motor) and self._motor.is_online()
        was_online = self._motor_online
        if online:
            if not was_online:
                self.log("✓ 执行系统已上线")
            self._motor_online = True
            self.lbl_motor.setText("执行系统: 在线")
            self.lbl_motor.setStyleSheet("color: #34c759; font-weight: 600;")
        else:
            if was_online:
                self.log("⚠ 执行系统离线！")
            self._motor_online = False
            self.lbl_motor.setText("执行系统: 离线")
            self.lbl_motor.setStyleSheet("color: #ff3b30; font-weight: 600;")



    def switch_mode(self, mode):
        """切换工作模式：点锡(solder)↔AOI，停止当前推理并更新UI状态"""
        if mode == self.current_mode:
            # QPushButton checkable默认可反选；点当前模式时强制保持选中, 避免三模式按钮全空
            self.btn_solder.setChecked(mode == "solder")
            self.btn_aoi.setChecked(mode == "aoi")
            self.btn_debug.setChecked(mode == "debug")
            return
        # 两种模式使用不同RKNN模型，切换时停止实时线程，避免模型错用
        if self.infer_thread and self.infer_thread.isRunning():
            self.stop_camera()
        self.current_mode = mode
        self.btn_solder.setChecked(mode == "solder")
        self.btn_aoi.setChecked(mode == "aoi")
        self.btn_debug.setChecked(mode == "debug")
        self.current_detections = None
        self.path_result = None
        self._update_mode_controls()
        if mode == "solder":
            self.log("⚙ 切换到 点锡模式")
        elif mode == "debug":
            self.log("🔧 切换到 调试模式")
            pass  # 显隐由_update_mode_controls处理
        else:
            self.log("⚙ 切换到 AOI检测模式")
            self._ensure_aoi_dir()

    def _update_mode_controls(self):
        """根据当前模式更新按钮可用性和文字(点锡/AOI/调试差异化)"""
        is_solder = self.current_mode == "solder"
        is_debug = self.current_mode == "debug"
        # 调试模式：隐藏整个右侧控制面板，显示调试面板
        if hasattr(self, '_right_widget'):
            self._right_widget.setVisible(not is_debug)
        if hasattr(self, '_debug_widget'):
            self._debug_widget.setVisible(is_debug)
        # 日志跟随模式：调试模式移到调试面板右栏，其他模式回right_widget
        if hasattr(self, 'log_group') and hasattr(self, '_debug_log_slot'):
            if is_debug:
                self._debug_log_slot.addWidget(self.log_group)
            elif self._right_layout.indexOf(self.log_group) < 0:
                self._right_layout.addWidget(self.log_group, 1)
        # 针尖校准相机: 调试模式显示切换条(由用户手动开/关), 离开则停预览并复位按钮
        if hasattr(self, '_tip_cam_bar'):
            self._tip_cam_bar.setVisible(is_debug)
            if not is_debug:
                self._close_tip_cam()
        # 标定按钮区: 仅调试模式可见, 离开则取消进行中的标定
        if hasattr(self, '_calib_bar'):
            self._calib_bar.setVisible(is_debug)
            if not is_debug and self._xy_calib_active:
                self._xy_calib_cancel()
        if is_debug:
            self.lbl_path.setText("调试模式")
            return
        self.btn_capture.setText("◎ 路径生成" if is_solder else "◎ 锁定当前帧")
        self.btn_capture.setEnabled(True)
        # btn_load: 点锡模式复用为"暂停"(待机禁用,执行时启用); AOI模式为"加载图片"
        if is_solder:
            self.btn_load.setText("⏸ 暂停")
            self.btn_load.setEnabled(False)
        else:
            self.btn_load.setText("⊞ 加载图片")
            self.btn_load.setEnabled(True)
        self.btn_resume.setVisible(False)
        self.btn_terminate.setVisible(False)
        self.btn_load.setVisible(True)
        self.btn_execute.setText("⚡ 执行点锡" if is_solder else "🔍 执行AOI检测")
        self.btn_execute.setEnabled(False)
        if is_solder:
            self.lbl_path.setText("路径: -- 点")
        else:
            self.lbl_path.setText("AOI: --")

    # ----------------------------------------------------------
    # 针尖校准相机预览 (仅调试模式, 显示在左侧视频区)
    # ----------------------------------------------------------
    def _on_tip_cam_changed(self, text):
        """针尖相机下拉切换: 若已打开则提示需重新打开新设备"""
        if self._tip_preview and self._tip_preview.isRunning():
            self.log(f"⚠ 已切换到相机 {text}, 请点[关闭]后重新[打开]以生效")

    def _open_tip_cam(self):
        """打开针尖校准相机预览(手动)。重复打开给提示。"""
        if self._tip_preview and self._tip_preview.isRunning():
            self.log("⚠ 针尖校准相机已打开")
            return
        try:
            cam_id = int(self.combo_tip_cam.currentText())
        except (ValueError, AttributeError):
            cam_id = 23
        self.log(f"📷 正在打开针尖校准相机 (/dev/video{cam_id}) ...")
        self._tip_preview = CameraPreviewThread(cam_id)
        self._tip_preview.frame_ready.connect(self._on_tip_frame)
        self._tip_preview.resolution_ready.connect(self._on_tip_resolution)
        self._tip_preview.open_failed.connect(self._on_tip_open_failed)
        self._tip_preview.start()
        self.btn_tip_open.setEnabled(False)
        self.btn_tip_close.setEnabled(True)
        self.combo_tip_cam.setEnabled(False)

    def _close_tip_cam(self):
        """关闭针尖校准相机预览(手动), 复位按钮与显示。"""
        was_open = bool(self._tip_preview and self._tip_preview.isRunning())
        if self._tip_preview:
            self._tip_preview.stop()
            self._tip_preview = None
        self.btn_tip_open.setEnabled(True)
        self.btn_tip_close.setEnabled(False)
        self.combo_tip_cam.setEnabled(True)
        if was_open:
            self.log("📷 针尖校准相机已关闭")
            self.video_label.setText("针尖校准相机已关闭")

    def _on_tip_open_failed(self, cam_id):
        """相机打开失败回调: 警告并复位按钮"""
        self.log(f"✗ 针尖校准相机 /dev/video{cam_id} 打开失败, 请检查连接")
        self._close_tip_cam()

    def _on_tip_resolution(self, w, h):
        """相机实际分辨率回调: 打到日志"""
        self.log(f"✓ 针尖校准相机已打开, 实际分辨率 {w}x{h}")

    def _on_tip_frame(self, frame):
        """针尖相机帧回调: BGR帧转pixmap显示到左侧视频区"""
        if self.current_mode != "debug":
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        label_size = self.video_label.size()
        pixmap = QPixmap.fromImage(qimg).scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setDisplayPixmap(pixmap)

    # ----------------------------------------------------------
    # XY 标定 (顶部相机像素 → 机床XY 仿射变换, 3点解算)
    # ----------------------------------------------------------
    def _grab_top_frame(self):
        """抓取顶部相机(video21)单帧BGR, 失败返回None。临时打开即用即放。"""
        try:
            cam_id = 21
            cap = cv2.VideoCapture(cam_id)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            frame = None
            for _ in range(8):
                ok, f = cap.read()
                if ok:
                    frame = f
            cap.release()
            # video21顶视相机畸变矫正
            frame = undistort_cam21(frame, cam_id)
            return frame
        except Exception as e:
            self.log(f"✗ 顶部相机抓帧异常: {e}")
            return None

    def _xy_calib_start(self):
        """开始XY标定: 停针尖预览, 抓顶部相机一帧冻结, 进入选点状态。"""
        if self._xy_calib_active:
            self.log("⚠ XY标定已在进行中")
            return
        self._close_tip_cam()  # 释放针尖相机, 避免占用
        self.log("▸ XY标定: 正在抓取顶部相机画面...")
        frame = self._grab_top_frame()
        if frame is None:
            self.log("✗ 顶部相机(/dev/video21)抓帧失败, 请检查连接")
            return
        self._xy_calib_active = True
        self._xy_calib_state = 'picking'
        self._xy_calib_frame = frame
        self._xy_calib_cur_px = None
        self._xy_calib_pairs = []
        self._xy_calib_redraw()
        self.btn_calib_cancel.setEnabled(True)
        self.btn_calib_lock.setEnabled(False)
        self.btn_calib_align.setEnabled(False)
        self.btn_calib_record.setEnabled(False)
        self._xy_calib_update_status()
        self.log("▸ 已冻结顶部画面, 请在图上点击选择特征点(可双指放大)")

    def _xy_calib_redraw(self):
        """在冻结帧上叠加已记录红点+当前候选绿点, 显示到左侧。"""
        if self._xy_calib_frame is None:
            return
        vis = self._xy_calib_frame.copy()
        # 已锁定/已记录的点画红色十字+编号
        for i, (u, v, X, Y) in enumerate(self._xy_calib_pairs):
            cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
            cv2.putText(vis, str(i + 1), (int(u) + 12, int(v) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # 当前候选点: locked画红, picking画绿
        if self._xy_calib_cur_px is not None:
            u, v = self._xy_calib_cur_px
            col = (0, 0, 255) if self._xy_calib_state in ('locked', 'aligning') else (0, 200, 0)
            cv2.drawMarker(vis, (int(u), int(v)), col, cv2.MARKER_CROSS, 40, 3)
        self.display_frame(vis)

    def _xy_calib_update_status(self):
        """刷新标定状态标签"""
        n = len(self._xy_calib_pairs)
        if not self._xy_calib_active:
            txt = "标定: 空闲"
            if self._xy_calib_M is not None:
                txt += " (已加载标定)"
        else:
            px = self._xy_calib_cur_px
            pxs = f" 像素({int(px[0])},{int(px[1])})" if px else ""
            txt = f"标定中[{self._xy_calib_state}] 已记录{n}/3{pxs}"
        self._lbl_calib_status.setText(txt)

    def _xy_calib_lock(self):
        """锁定当前候选点: 绿点→红点, 显示像素坐标, 启用[手动标定]。"""
        if self._xy_calib_state != 'picking' or self._xy_calib_cur_px is None:
            self.log("⚠ 请先在图上点击一个点再锁定")
            return
        self._xy_calib_state = 'locked'
        self.btn_calib_lock.setEnabled(False)
        self.btn_calib_align.setEnabled(True)
        u, v = self._xy_calib_cur_px
        self.log(f"▸ 已锁定点 像素=({int(u)},{int(v)}), 请用右侧面板jog针尖到该物理位置")
        self._xy_calib_redraw()
        self._xy_calib_update_status()

    def _xy_calib_align(self):
        """打开针尖相机(video23)供精对准, 启用[确认记录]。"""
        if self._xy_calib_state != 'locked':
            self.log("⚠ 请先锁定点再手动标定")
            return
        self._xy_calib_state = 'aligning'
        # 借用针尖相机预览(切到video23)
        try:
            self.combo_tip_cam.setCurrentText("23")
        except Exception:
            pass
        self._open_tip_cam()
        self.btn_calib_align.setEnabled(False)
        self.btn_calib_record.setEnabled(True)
        self.log("▸ 针尖相机已开, 对准后按[确认记录]记录当前机床XY")

    def _xy_calib_record(self):
        """记录当前机床XY与锁定像素配对; 满3点则解算保存。"""
        if self._xy_calib_state != 'aligning' or self._xy_calib_cur_px is None:
            self.log("⚠ 当前不可记录")
            return
        u, v = self._xy_calib_cur_px
        X, Y = self._current_x, self._current_y
        self._xy_calib_pairs.append((float(u), float(v), float(X), float(Y)))
        self.log(f"✓ 记录点{len(self._xy_calib_pairs)}: 像素({int(u)},{int(v)}) ↔ 机床({X:.1f},{Y:.1f})")
        # 关针尖相机, 切回冻结顶图
        self._close_tip_cam()
        self._xy_calib_cur_px = None
        if len(self._xy_calib_pairs) >= 3:
            self._xy_calib_solve_save()
        else:
            self._xy_calib_state = 'picking'
            self.btn_calib_record.setEnabled(False)
            self._xy_calib_redraw()
            self._xy_calib_update_status()
            self.log(f"▸ 请选择第 {len(self._xy_calib_pairs)+1} 个点")

    def _xy_calib_solve_save(self):
        """用3组点解算仿射矩阵(像素→机床), 计算残差, 存盘。"""
        import numpy as np
        pts = self._xy_calib_pairs
        src = np.array([[u, v] for (u, v, X, Y) in pts], dtype=np.float32)
        dst = np.array([[X, Y] for (u, v, X, Y) in pts], dtype=np.float32)
        try:
            M = cv2.getAffineTransform(src, dst)  # 2x3
        except Exception as e:
            self.log(f"✗ 解算失败: {e}, 请重置重标")
            self._xy_calib_cancel()
            return
        # 残差: 把src点用M映射回机床坐标, 与dst比
        res = []
        for (u, v, X, Y) in pts:
            mx = M[0, 0]*u + M[0, 1]*v + M[0, 2]
            my = M[1, 0]*u + M[1, 1]*v + M[1, 2]
            res.append(((mx-X)**2 + (my-Y)**2) ** 0.5)
        max_res = max(res)
        self._xy_calib_M = M
        self._save_xy_calib(M, pts, max_res)
        self.log(f"✓ XY标定完成! 仿射矩阵已保存, 最大残差 {max_res:.2f}mm")
        # 收尾
        self._xy_calib_active = False
        self._xy_calib_state = 'idle'
        for b in (self.btn_calib_lock, self.btn_calib_align, self.btn_calib_record, self.btn_calib_cancel):
            b.setEnabled(False)
        self._xy_calib_update_status()

    def _xy_calib_cancel(self):
        """取消/重置XY标定, 清状态。"""
        self._xy_calib_active = False
        self._xy_calib_state = 'idle'
        self._xy_calib_frame = None
        self._xy_calib_cur_px = None
        self._xy_calib_pairs = []
        self._close_tip_cam()
        for b in (self.btn_calib_lock, self.btn_calib_align, self.btn_calib_record, self.btn_calib_cancel):
            b.setEnabled(False)
        self._xy_calib_update_status()
        self.log("▸ XY标定已取消/重置")

    def _save_xy_calib(self, M, pts, max_res):
        """保存标定矩阵+原始点+时间戳到json。"""
        import json
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            data = {
                "affine_matrix": [[float(M[0, 0]), float(M[0, 1]), float(M[0, 2])],
                                  [float(M[1, 0]), float(M[1, 1]), float(M[1, 2])]],
                "points": [{"u": u, "v": v, "X": X, "Y": Y} for (u, v, X, Y) in pts],
                "max_residual_mm": float(max_res),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(XY_CALIB_PATH, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"✗ 标定文件保存失败: {e}")

    def _load_xy_calib(self):
        """上电加载已保存的XY标定矩阵(若存在)。"""
        import json
        try:
            if os.path.exists(XY_CALIB_PATH):
                with open(XY_CALIB_PATH) as f:
                    data = json.load(f)
                import numpy as np
                self._xy_calib_M = np.array(data["affine_matrix"], dtype=np.float64)
                print(f"[xy_calib] loaded: {data.get('timestamp')}, residual={data.get('max_residual_mm')}")
        except Exception as e:
            print(f"[xy_calib] load failed: {e}")

    def pixel_to_machine(self, u, v):
        """用标定矩阵把顶部相机像素(u,v)换算为机床(X,Y)。未标定返回None。"""
        if self._xy_calib_M is None:
            return None
        M = self._xy_calib_M
        X = M[0, 0]*u + M[0, 1]*v + M[0, 2]
        Y = M[1, 0]*u + M[1, 1]*v + M[1, 2]
        return (float(X), float(Y))

    # ----------------------------------------------------------
    # XY测试: 在顶图选点→标定矩阵换算机床XY→GO移动(仅XY,不动Z)
    # ----------------------------------------------------------
    def _xytest_pick_toggle(self):
        """选点/锁定点 切换。选点:抓顶图冻结进入picking; 锁定:换算XY,按钮变回选点可重选。"""
        if self._xytest_state == 'idle':
            if self._xy_calib_M is None:
                self.log("⚠ 请先完成XY标定再测试")
                return
            # 若标定流程在进行中, 不抢占
            if self._xy_calib_active:
                self.log("⚠ 请先结束XY标定再测试")
                return
            self.log("▸ XY测试: 抓取顶部画面...")
            frame = self._grab_top_frame()
            if frame is None:
                self.log("✗ 顶部相机抓帧失败")
                return
            self._xytest_state = 'picking'
            self._xytest_frame = frame
            self._xytest_px = None
            self._xytest_xy = None
            self._lbl_xytest.setText("XY: 点图选点")
            self.display_frame(frame)
            self._btn_xytest_pick.setText("锁定点")
            self.log("▸ 请在图上点击一个点(可放大)")
        elif self._xytest_state == 'picking':
            if self._xytest_px is None:
                self.log("⚠ 请先在图上点击一个点")
                return
            u, v = self._xytest_px
            xy = self.pixel_to_machine(u, v)
            if xy is None:
                self.log("⚠ 未标定, 无法换算")
                return
            self._xytest_xy = xy
            self._lbl_xytest.setText(f"XY: {xy[0]:.1f}, {xy[1]:.1f}")
            self.log(f"▸ 锁定 像素({u},{v}) → 机床XY({xy[0]:.1f},{xy[1]:.1f}), 按GO移动")
            self._xytest_state = 'idle'
            self._btn_xytest_pick.setText("选点")

    def _xytest_redraw(self):
        """在冻结顶图上画选中像素点(picking绿/已锁定红)+换算坐标, 显示到左侧。"""
        if self._xytest_frame is None or self._xytest_px is None:
            return
        vis = self._xytest_frame.copy()
        u, v = self._xytest_px
        col = (0, 0, 255) if self._xytest_state == 'idle' else (0, 200, 0)
        cv2.drawMarker(vis, (int(u), int(v)), col, cv2.MARKER_CROSS, 40, 3)
        self.display_frame(vis)

    def _xytest_go(self):
        """移动到换算出的XY(仅XY, 复用0x01, 不动Z)。"""
        if self._xytest_xy is None:
            self.log("⚠ 请先选点并锁定再GO")
            return
        x, y = self._xytest_xy
        x = max(0.0, min(2475.0, x))
        y = max(0.0, min(2475.0, y))
        self._current_x, self._current_y = x, y
        self._update_coord_display()
        self._set_motion_state("moving")
        self.log(f"▸ XY测试移动 → X={x:.1f} Y={y:.1f} (Z不动)")
        self._send_cmd(0x01, x, y)
        self._homed = False

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
        self.infer_thread.conf_thresh = float(self.spin_conf.value())
        self.infer_thread.use_remote = self._use_remote
        self.infer_thread.remote_client = self._remote_client
        self.infer_thread.result_ready.connect(self.on_result)
        self.infer_thread.remote_error.connect(lambda m: self.log(f"⚠ 外部推理失败: {m}"))
        self.infer_thread.set_camera(cam_id)
        self.infer_thread.start()
        self._frozen = False
        self._edit_mode = False
        self._current_x = 0.0
        self._current_y = 0.0
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
                self._selection_mask = [int(d[2]) == 0 for d in self.current_detections]  # 默认只选pad(class=0),hole/qfn不选
            self._edit_mode = True
            self._redraw_edit_frame()
            self.log("◎ 编辑模式：点击框可取消/恢复选中")


    def _redraw_edit_frame(self):
        """在编辑模式下重绘帧：选中的加蒙版，未选中的只有边框"""
        if self.current_frame is None:
            return
        vis = self.current_frame.copy()
        overlay = vis.copy()
        has_mask = False
        # 先在overlay上画所有选中框的填充蒙版(只copy一次)
        for i, det in enumerate(self.current_detections):
            if self._selection_mask[i]:
                x1, y1, x2, y2 = [int(v) for v in det[0]]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 160, 0), -1)
                has_mask = True
        if has_mask:
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        # 再画所有边框
        for i, det in enumerate(self.current_detections):
            x1, y1, x2, y2 = [int(v) for v in det[0]]
            if self._selection_mask[i]:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        self.display_frame(vis)

    def _on_image_clicked(self, x_ratio, y_ratio):
        """图像区域被点击（坐标为相对于图像的0~1比例）"""
        # XY标定选点: picking状态下点击设候选绿点
        if self._xy_calib_active and self._xy_calib_state == 'picking' and self._xy_calib_frame is not None:
            h, w = self._xy_calib_frame.shape[:2]
            u, v = int(x_ratio * w), int(y_ratio * h)
            self._xy_calib_cur_px = (u, v)
            self.btn_calib_lock.setEnabled(True)
            self.log(f"▸ 候选点 像素=({u},{v}), 按[锁定点]确认")
            self._xy_calib_redraw()
            self._xy_calib_update_status()
            return
        # XY测试选点: picking状态下点击设选中像素点
        if self._xytest_state == 'picking' and self._xytest_frame is not None:
            h, w = self._xytest_frame.shape[:2]
            u, v = int(x_ratio * w), int(y_ratio * h)
            self._xytest_px = (u, v)
            self._lbl_xytest.setText(f"像素:({u},{v}) 按锁定")
            self.log(f"▸ XY测试候选 像素=({u},{v}), 按[锁定点]换算")
            self._xytest_redraw()
            return
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
            # 记录生成路径时的显示裁剪x偏移, 执行时用于把显示坐标还原为全图坐标(与XY标定对齐)
            self._path_disp_offset_x = getattr(self.infer_thread, 'disp_offset_x', 0) if self.infer_thread else 0

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

    def _toggle_infer_source(self):
        """切换本地/外部推理。外部仅在推理系统在线时可切, 否则报警不切。"""
        if not self._use_remote:
            # 本地 → 外部: 需在线
            if not self._remote_online:
                self.log("⚠ 推理系统离线, 无法切换到外部推理")
                return
            self._use_remote = True
            self.btn_infer_src.setText("外部推理")
            self.btn_infer_src.setStyleSheet("background: #ff9500; color: white; font-weight: 600;")
            self.log("✓ 已切换到外部推理")
        else:
            # 外部 → 本地
            self._use_remote = False
            self.btn_infer_src.setText("本地推理")
            self.btn_infer_src.setStyleSheet("")
            self.log("✓ 已切换到本地推理")
        # 同步到推理线程
        if self.infer_thread is not None:
            self.infer_thread.use_remote = self._use_remote
            self.infer_thread.remote_client = self._remote_client

    def _on_remote_status_changed(self, online):
        """推理系统在线状态变化时更新UI(由后台线程信号触发, 运行在主线程)"""
        self._remote_online = online
        if online:
            self.lbl_infersys.setText("推理系统: 在线")
            self.lbl_infersys.setStyleSheet("color: #34c759; font-weight: 600;")
        else:
            self.lbl_infersys.setText("推理系统: 离线")
            self.lbl_infersys.setStyleSheet("color: #ff3b30; font-weight: 600;")

    def _on_remote_status_tick(self, online):
        """每次探测结果: 若离线且当前处于外部推理则自动回退本地"""
        if not online and self._use_remote:
            self._fallback_local("推理系统掉线")

    def _fallback_local(self, reason=""):
        """从外部推理自动回退到本地推理。"""
        self._use_remote = False
        self.btn_infer_src.setText("本地推理")
        self.btn_infer_src.setStyleSheet("")
        if self.infer_thread is not None:
            self.infer_thread.use_remote = False
        self.log(f"⚠ {reason}, 已自动切回本地推理")

    def _infer_once(self, frame, model_path):
        """对单帧执行一次推理(阻塞)，返回(detections, elapsed_ms)。
        外部推理模式走远程, 否则本地RKNN。远程失败则报警返回空。"""
        t0 = time.time()
        # frame裁剪中心正方形ROI送推理
        fh, fw = frame.shape[:2]
        side = min(fh, fw)
        rx, ry = (fw - side) // 2, (fh - side) // 2
        roi = frame[ry:ry+side, rx:rx+side]

        if self._use_remote and self._remote_client is not None:
            try:
                bboxes, scores, class_ids = self._remote_client.infer(roi)
            except Exception as e:
                self.log(f"⚠ 外部推理失败: {e}")
                return [], (time.time() - t0) * 1000
        else:
            from rknnlite.api import RKNNLite
            from infer import infer
            rknn = RKNNLite()
            rknn.load_rknn(model_path)
            rknn.init_runtime()
            bboxes, scores, class_ids = infer(rknn, roi)
            try:
                rknn.release()
            except Exception:
                pass
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] += rx
            bboxes[:, [1, 3]] += ry
        elapsed = (time.time() - t0) * 1000
        return list(zip(bboxes, scores, class_ids)) if len(bboxes) > 0 else [], elapsed

    # 类别名与颜色 (0=pad, 1=hole, 2=qfn)
    CLASS_NAMES = ("pad", "hole", "qfn")
    CLASS_COLORS = ((0, 255, 0), (0, 165, 255), (255, 128, 0))  # pad绿, hole橙, qfn蓝

    def _draw_detections(self, frame, detections, color=None, prefix=None):
        """在帧上绘制检测框和标签。color/prefix为None时按类别自动着色+真实类别名。"""
        vis = frame.copy()
        for bbox, score, cls_id in detections:
            x1, y1, x2, y2 = map(int, bbox)
            ci = int(cls_id)
            if color is None:
                c = self.CLASS_COLORS[ci] if 0 <= ci < len(self.CLASS_COLORS) else (0, 255, 0)
                name = self.CLASS_NAMES[ci] if 0 <= ci < len(self.CLASS_NAMES) else str(ci)
            else:
                c = color
                name = (prefix or "") + (self.CLASS_NAMES[ci] if 0 <= ci < len(self.CLASS_NAMES) else str(ci))
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            label = f"{name} {score:.2f}"
            cv2.putText(vis, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
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
        """执行点锡动作：XY到点→Z下降固定820步→挤锡→Z回抬820步→下一点。"""
        if self.current_mode != "solder":
            self.execute_aoi()
            return
        if not self.path_result or not self.path_result.get('points'):
            self.log("⚠ 当前无可用路径，请先点击「路径生成」")
            return
        if self._motor is None or not self._motor.is_online():
            self.log("⚠ 执行系统离线，无法执行点锡")
            return
        if self._xy_calib_M is None:
            self.log("⚠ 请先完成XY标定，再执行点锡")
            return

        exec_points = []
        for i, pt in enumerate(self.path_result['points']):
            try:
                u, v = float(pt['x']), float(pt['y'])
            except Exception:
                self.log(f"⚠ 路径点{i}格式错误，跳过: {pt}")
                continue
            # 还原显示裁剪x偏移→全图坐标(XY标定基于全图1920坐标)
            u += getattr(self, '_path_disp_offset_x', 0)
            xy = self.pixel_to_machine(u, v)
            if xy is None:
                self.log("⚠ XY标定无效，无法换算机床坐标")
                return
            exec_points.append({"u": u, "v": v, "x": xy[0], "y": xy[1], "raw": pt})

        if not exec_points:
            self.log("⚠ 无有效点锡点")
            return

        # 预构建命令队列: 开始回零→[XY→Z下降到接触→挤锡→Z抬到安全位]×N→结束回零
        # Z命令存目标绝对位置(z_abs), 发送时按当前实际Z实时算delta, 暂停回零后续跑也不会错乱
        total = len(exec_points)
        cmds = [{'label': '开始回零', 'cmd': 0x03, 'args': (),
                 'wait': SOLDER_HOME_MIN_WAIT_MS, 'timeout': SOLDER_HOME_TIMEOUT_MS}]
        for i, pt in enumerate(exec_points):
            cmds.append({'label': f"点{i+1}/{total} XY→({pt['x']:.1f},{pt['y']:.1f})",
                         'cmd': 0x01, 'args': (pt['x'], pt['y']),
                         'wait': SOLDER_XY_MIN_WAIT_MS, 'timeout': SOLDER_STEP_TIMEOUT_MS})
            cmds.append({'label': f'  Z下降到接触({SOLDER_Z_DOWN_STEPS})', 'cmd': 0x06, 'z_abs': SOLDER_Z_DOWN_STEPS,
                         'wait': SOLDER_Z_DOWN_MIN_WAIT_MS, 'timeout': SOLDER_STEP_TIMEOUT_MS})
            cmds.append({'label': '  挤锡', 'cmd': 0x07, 'args': (SOLDER_SQUEEZE_COUNT,),
                         'wait': SOLDER_SQUEEZE_MIN_WAIT_MS, 'timeout': SOLDER_STEP_TIMEOUT_MS})
            cmds.append({'label': f'  Z抬到{SOLDER_Z_LIFT_POS}', 'cmd': 0x06, 'z_abs': SOLDER_Z_LIFT_POS,
                         'wait': SOLDER_Z_UP_MIN_WAIT_MS, 'timeout': SOLDER_STEP_TIMEOUT_MS,
                         'pt_end': i})
        cmds.append({'label': '结束回零', 'cmd': 0x03, 'args': (),
                     'wait': SOLDER_HOME_MIN_WAIT_MS, 'timeout': SOLDER_HOME_TIMEOUT_MS})

        self._exec_cmds = cmds
        self._exec_ci = 0
        self._exec_waiting = False
        self._exec_cmd_ts = 0.0
        self._exec_started_at = time.time()
        self._exec_pause_req = False
        self._exec_paused = False
        self._exec_cur_z = 0
        self.progress_bar.setMaximum(len(cmds))
        self.progress_bar.setValue(0)
        self.btn_execute.setEnabled(False)
        # 进入执行态: 暂停按钮可用, 继续/终止隐藏
        self.btn_load.setText("⏸ 暂停")
        self.btn_load.setEnabled(True)
        self.btn_load.setVisible(True)
        self.btn_resume.setVisible(False)
        self.btn_terminate.setVisible(False)

        self._exec_timer = QTimer()
        self._exec_timer.timeout.connect(self._exec_step)
        self._exec_timer.start(SOLDER_TIMER_INTERVAL_MS)
        self.log(f"⚙ 开始执行点锡: {total}点，前后各回零，点间Z抬至{SOLDER_Z_LIFT_POS}")

    def _exec_abort(self, reason):
        """停止点锡状态机并恢复按钮。"""
        try:
            if hasattr(self, '_exec_timer') and self._exec_timer:
                self._exec_timer.stop()
        except Exception:
            pass
        self._exec_paused = False
        self._exec_pause_req = False
        self._exec_pause_homing = False
        self._solder_reset_buttons()
        self.btn_execute.setEnabled(True)
        self.log(f"✗ 点锡执行中止: {reason}")

    def _solder_reset_buttons(self):
        """点锡按钮恢复到待机态: 仅显示禁用的'暂停', 隐藏继续/终止。"""
        self.btn_resume.setVisible(False)
        self.btn_terminate.setVisible(False)
        self.btn_load.setVisible(True)
        self.btn_load.setText("⏸ 暂停")
        self.btn_load.setEnabled(False)

    def _on_btn_load_clicked(self):
        """btn_load点击分派: 点锡模式=暂停; AOI模式=加载图片。"""
        if self.current_mode == "solder":
            self._on_solder_pause()
        else:
            self.load_image()

    def _on_solder_pause(self):
        """暂停: 仅置标志, 状态机会在完成当前点(pt_end)后回零并真正停下。"""
        if not getattr(self, '_exec_timer', None) or not self._exec_timer.isActive():
            return
        if self._exec_pause_req or self._exec_paused:
            return
        self._exec_pause_req = True
        self.btn_load.setEnabled(False)
        self.btn_load.setText("⏸ 暂停中…")
        self.log("⏸ 已请求暂停，完成当前点后回零停下")

    def _on_solder_resume(self):
        """继续: 从暂停处接着点。"""
        if not self._exec_paused:
            return
        self._exec_paused = False
        self.btn_resume.setVisible(False)
        self.btn_terminate.setVisible(False)
        self.btn_load.setVisible(True)
        self.btn_load.setText("⏸ 暂停")
        self.btn_load.setEnabled(True)
        self.log(f"▶ 继续点锡 (从第{self._exec_ci+1}/{len(self._exec_cmds)}步)")
        if not getattr(self, '_exec_timer', None):
            self._exec_timer = QTimer()
            self._exec_timer.timeout.connect(self._exec_step)
        self._exec_timer.start(SOLDER_TIMER_INTERVAL_MS)

    def _on_solder_terminate(self):
        """终止: 直接结束(暂停已含回零, 不再回零)。"""
        if not self._exec_paused:
            return
        try:
            if getattr(self, '_exec_timer', None):
                self._exec_timer.stop()
        except Exception:
            pass
        self._exec_paused = False
        self._exec_pause_req = False
        self._solder_reset_buttons()
        self.btn_execute.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log("■ 点锡已终止")


    def _exec_motor_busy(self):
        """读取motor busy状态，兼容旧版motor_control无is_busy。"""
        try:
            return bool(self._motor.is_busy())
        except Exception:
            return False

    def _exec_wait_done(self):
        """正在等待的命令(_exec_wait_cmd): busy未释放或未达最小等待则继续等; 超时则中止。"""
        if not self._exec_waiting:
            return True
        cur = getattr(self, '_exec_wait_cmd', None) or {}
        elapsed_ms = (time.time() - self._exec_cmd_ts) * 1000
        if elapsed_ms > cur.get('timeout', SOLDER_STEP_TIMEOUT_MS):
            self._exec_abort(f"等待[{cur.get('label','?').strip()}]完成超时")
            return False
        if elapsed_ms < cur.get('wait', 0):
            return False
        if self._exec_motor_busy():
            return False
        self._exec_waiting = False
        return True

    def _exec_step(self):
        """点锡执行状态机: 顺序发送预构建命令队列, 每条等待完成后再发下一条。"""
        if self._motor is None or not self._motor.is_online():
            self._exec_abort("执行系统离线")
            return
        if not self._exec_wait_done():
            return

        # 暂停检测: 上一条命令是某点的收尾(pt_end)且收到暂停请求 → 回零后停下
        if self._exec_pause_req and not self._exec_paused and self._exec_ci > 0:
            prev = self._exec_cmds[self._exec_ci - 1]
            if 'pt_end' in prev:
                if not getattr(self, '_exec_pause_homing', False):
                    # 先发一次回零, 等其完成再真正停
                    self._exec_pause_homing = True
                    self.log("⏸ 暂停: 当前点已完成，回零中…")
                    if self._send_cmd(0x03) is False:
                        self._exec_abort("暂停回零发送失败")
                        return
                    self._exec_waiting = True
                    self._exec_cmd_ts = time.time()
                    self._exec_wait_cmd = {'label': '暂停回零', 'wait': SOLDER_HOME_MIN_WAIT_MS,
                                           'timeout': SOLDER_HOME_TIMEOUT_MS}
                    return
                else:
                    # 回零完成, 进入暂停态
                    self._exec_pause_homing = False
                    self._exec_pause_req = False
                    self._exec_cur_z = 0
                    self._exec_paused = True
                    self._exec_timer.stop()
                    self.btn_load.setVisible(False)
                    self.btn_resume.setVisible(True)
                    self.btn_terminate.setVisible(True)
                    self.log(f"⏸ 已暂停 (已完成{self._exec_ci}/{len(self._exec_cmds)}步)，点继续接着点，或终止")
                    return

        if self._exec_ci >= len(self._exec_cmds):
            self._exec_timer.stop()
            self.progress_bar.setValue(self.progress_bar.maximum())
            self._exec_paused = False
            self._exec_pause_req = False
            self._solder_reset_buttons()
            self.btn_execute.setEnabled(True)
            elapsed = time.time() - getattr(self, '_exec_started_at', time.time())
            self.log(f"✓ 点锡执行完成，用时{elapsed:.1f}s")
            return

        c = self._exec_cmds[self._exec_ci]
        self.log(f"▸ {c['label']}")
        # Z命令用绝对目标实时换算相对步数, 兼容暂停回零后继续
        args = c.get('args', ())
        next_z = None
        if c.get('cmd') == 0x06 and 'z_abs' in c:
            cur_z = getattr(self, '_exec_cur_z', 0)
            next_z = int(c['z_abs'])
            args = (next_z - cur_z,)
            self.log(f"  Z: {cur_z} → {next_z} (delta {args[0]})")
        ok = self._send_cmd(c['cmd'], *args)
        if ok is False:
            self._exec_abort(f"发送[{c['label'].strip()}]失败")
            return
        if c.get('cmd') == 0x03:
            self._exec_cur_z = 0
        elif next_z is not None:
            self._exec_cur_z = next_z
        self._exec_ci += 1
        self._exec_waiting = True
        self._exec_cmd_ts = time.time()
        self._exec_wait_cmd = c
        self.progress_bar.setValue(self._exec_ci)

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
            vis = self._draw_detections(frame, detections)

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


    # ============================================================
    # 调试模式面板与通信
    # ============================================================
    def _create_debug_panel(self):
        """创建调试模式面板(双栏布局)：左栏=XY/Z控制+步进，右栏=目标坐标+操作+状态+激光"""
        from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLineEdit, QSizePolicy
        
        panel = QWidget()
        panel.setFixedWidth(S(560))
        two_col = QHBoxLayout(panel)
        two_col.setSpacing(S(10))
        two_col.setContentsMargins(S(6), S(6), S(6), S(6))
        
        # ===== 左栏 =====
        left_col = QVBoxLayout()
        left_col.setSpacing(S(8))
        
        # 状态指示
        status_grp = QGroupBox("系统状态")
        status_lay = QHBoxLayout(status_grp)
        self._dbg_status_led = QLabel("⚪")
        self._dbg_status_led.setStyleSheet(f"font-size: {S(18)}px;")
        self._dbg_status_txt = QLabel("未连接")
        self._dbg_status_txt.setStyleSheet(f"font-size: {S(15)}px; font-weight: bold;")
        status_lay.addWidget(self._dbg_status_led)
        status_lay.addStretch()
        status_lay.addWidget(self._dbg_status_txt)
        status_lay.addStretch()
        left_col.addWidget(status_grp)
        
        # 实时坐标
        coord_grp = QGroupBox("实时坐标")
        coord_lay = QHBoxLayout(coord_grp)
        coord_lay.setContentsMargins(S(10), S(10), S(10), S(10))
        self._lbl_coord_x = QLabel("X: 0.0")
        self._lbl_coord_y = QLabel("Y: 0.0")
        self._lbl_coord_z = QLabel("Z: --")
        coord_colors = {self._lbl_coord_x: "#ff3b30",   # X 红
                        self._lbl_coord_y: "#34c759",   # Y 绿
                        self._lbl_coord_z: "#007aff"}   # Z 蓝
        for lbl, col in coord_colors.items():
            lbl.setStyleSheet(f"font-size: {S(15)}px; font-weight: bold; font-family: monospace; color: {col};")
            coord_lay.addWidget(lbl)
        coord_grp.setMinimumHeight(S(62))
        left_col.addWidget(coord_grp)
        
        # XY方向控制(紧凑)
        xy_grp = QGroupBox("XY 移动")
        xy_grid = QGridLayout(xy_grp)
        xy_grid.setSpacing(S(4))
        xy_grid.setContentsMargins(S(6), S(4), S(6), S(4))
        btn_style = f"font-size: {S(14)}px; min-height: {S(34)}px; max-height: {S(34)}px; min-width: {S(34)}px; border-radius: {S(6)}px; background: #e5e5ea;"
        btn_up = QPushButton("▲")
        btn_down = QPushButton("▼")
        btn_left = QPushButton("◀")
        btn_right = QPushButton("▶")
        for b in (btn_up, btn_down, btn_left, btn_right):
            b.setStyleSheet(btn_style)
        xy_grid.addWidget(btn_up, 0, 1)
        xy_grid.addWidget(btn_left, 1, 0)
        xy_grid.addWidget(btn_right, 1, 2)
        xy_grid.addWidget(btn_down, 2, 1)
        btn_up.clicked.connect(lambda: self._cmd_xy_move(1, 0))
        btn_down.clicked.connect(lambda: self._cmd_xy_move(-1, 0))
        btn_left.clicked.connect(lambda: self._cmd_xy_move(0, 1))
        btn_right.clicked.connect(lambda: self._cmd_xy_move(0, -1))
        left_col.addWidget(xy_grp)
        
        # Z轴控制
        z_grp = QGroupBox("Z 轴")
        z_lay = QHBoxLayout(z_grp)
        btn_z_up = QPushButton("▲ Z上")
        btn_z_down = QPushButton("▼ Z下")
        for b in (btn_z_up, btn_z_down):
            b.setStyleSheet(f"font-size: {S(11)}px; min-height: {S(38)}px; background: #e5e5ea; border-radius: {S(8)}px;")
        btn_z_up.clicked.connect(lambda: self._cmd_z_move(1))
        btn_z_down.clicked.connect(lambda: self._cmd_z_move(-1))
        z_lay.addWidget(btn_z_up)
        z_lay.addWidget(btn_z_down)
        left_col.addWidget(z_grp)
        
        # 步进量选择 (2x4 八档)
        step_grp = QGroupBox("步进量 (°)")
        step_grid = QGridLayout(step_grp)
        step_grid.setSpacing(S(4))
        step_grid.setContentsMargins(S(6), S(4), S(6), S(4))
        self._step_btns = []
        step_vals = [1, 5, 10, 50, 100, 200, 500, 1000]
        for idx, val in enumerate(step_vals):
            b = QPushButton(str(val))
            b.setCheckable(True)
            b.setStyleSheet(f"font-size: {S(12)}px; font-weight: bold; min-height: {S(30)}px; border-radius: {S(6)}px;")
            b.clicked.connect(lambda checked, v=val: self._set_step_size(v))
            step_grid.addWidget(b, idx // 4, idx % 4)
            self._step_btns.append((b, val))
        self._step_btns[3][0].setChecked(True)  # 默认选中 50
        self._step_size = 50.0
        left_col.addWidget(step_grp)
        
        left_col.addStretch()
        two_col.addLayout(left_col)
        
        # ===== 右栏 =====
        right_col = QVBoxLayout()
        right_col.setSpacing(S(8))
        
        # 目标坐标输入
        goto_grp = QGroupBox("运动到坐标")
        goto_lay = QHBoxLayout(goto_grp)
        self._input_x = QLineEdit("0")
        self._input_y = QLineEdit("0")
        self._input_z = QLineEdit("0")
        self._input_x.setFixedWidth(S(48))
        self._input_y.setFixedWidth(S(48))
        self._input_z.setFixedWidth(S(48))
        for ed in (self._input_x, self._input_y, self._input_z):
            ed.setAlignment(Qt.AlignCenter)
        self._input_x.setStyleSheet(f"font-size: {S(11)}px;")
        self._input_y.setStyleSheet(f"font-size: {S(11)}px;")
        self._input_z.setStyleSheet(f"font-size: {S(11)}px;")
        btn_goto = QPushButton("Go")
        btn_goto.setStyleSheet(f"font-size: {S(11)}px; min-height: {S(32)}px; background: #007aff; color: white; border: none; border-radius: {S(8)}px; padding: 0 {S(10)}px;")
        btn_goto.clicked.connect(self._cmd_goto_xy)
        goto_lay.addWidget(QLabel("X:"))
        goto_lay.addWidget(self._input_x)
        goto_lay.addWidget(QLabel("Y:"))
        goto_lay.addWidget(self._input_y)
        goto_lay.addWidget(QLabel("Z:"))
        goto_lay.addWidget(self._input_z)
        goto_lay.addWidget(btn_goto)
        right_col.addWidget(goto_grp)

        # XY测试: 验证标定准不准. 左侧显示换算的机床XY, 右侧[选点/锁定点]+[GO](只动XY,不动Z)
        xytest_grp = QGroupBox("XY测试 (验证标定)")
        xytest_lay = QHBoxLayout(xytest_grp)
        xytest_lay.setSpacing(S(6))
        self._lbl_xytest = QLabel("XY: --")
        self._lbl_xytest.setStyleSheet(f"font-size: {S(12)}px; font-weight: bold; color: #5856d6;")
        self._btn_xytest_pick = QPushButton("选点")
        self._btn_xytest_pick.setFixedHeight(S(32))
        self._btn_xytest_pick.setStyleSheet(f"font-size: {S(11)}px; background: #34c759; color: white; border: none; border-radius: {S(6)}px;")
        self._btn_xytest_pick.clicked.connect(self._xytest_pick_toggle)
        self._btn_xytest_go = QPushButton("GO")
        self._btn_xytest_go.setFixedHeight(S(32))
        self._btn_xytest_go.setStyleSheet(f"font-size: {S(11)}px; background: #007aff; color: white; border: none; border-radius: {S(6)}px;")
        self._btn_xytest_go.clicked.connect(self._xytest_go)
        xytest_lay.addWidget(self._lbl_xytest, 1)
        xytest_lay.addWidget(self._btn_xytest_pick, 1)
        xytest_lay.addWidget(self._btn_xytest_go, 1)
        right_col.addWidget(xytest_grp)
        
        # 操作按钮: 左侧回零/挤锡上下叠放, 右侧急停单独加大(更易触发)
        action_grp = QGroupBox("操作")
        action_lay = QHBoxLayout(action_grp)
        action_lay.setSpacing(S(8))
        # 左列: 回零 + 挤锡
        left_btns = QVBoxLayout()
        left_btns.setSpacing(S(8))
        btn_home = QPushButton("🏠 回零")
        btn_home.setFixedHeight(S(40))
        btn_home.setStyleSheet(f"font-size: {S(11)}px; background: #34c759; color: white; border: none; border-radius: {S(8)}px;")
        btn_home.clicked.connect(self._cmd_home)
        btn_squeeze = QPushButton("💧 挤锡")
        btn_squeeze.setFixedHeight(S(40))
        btn_squeeze.setStyleSheet(f"font-size: {S(11)}px; background: #ff9500; color: white; border: none; border-radius: {S(8)}px;")
        btn_squeeze.clicked.connect(self._cmd_squeeze)
        left_btns.addWidget(btn_home)
        left_btns.addWidget(btn_squeeze)
        action_lay.addLayout(left_btns, 1)
        # 右列: 急停(加大, 占满整高)
        btn_estop = QPushButton("🛑\n急停")
        btn_estop.setMinimumHeight(S(88))
        btn_estop.setStyleSheet(f"font-size: {S(16)}px; font-weight: bold; background: #ff3b30; color: white; border: none; border-radius: {S(10)}px;")
        btn_estop.clicked.connect(self._cmd_estop)
        action_lay.addWidget(btn_estop, 1)
        right_col.addWidget(action_grp)
        
        # 激光测距(左半) + 绝对零点校准(右半)
        laser_grp = QGroupBox("激光测距 / 零点校准")
        laser_lay = QHBoxLayout(laser_grp)
        laser_lay.setSpacing(S(8))
        self._lbl_laser = QLabel("距离: -- mm")
        self._lbl_laser.setStyleSheet(f"font-size: {S(12)}px; font-weight: bold;")
        laser_lay.addWidget(self._lbl_laser, 1)
        self._btn_calib = QPushButton("绝对零点校准")
        self._btn_calib.setStyleSheet(f"font-size: {S(11)}px; min-height: {S(36)}px; background: #5856d6; color: white; border: none; border-radius: {S(8)}px;")
        self._btn_calib.setEnabled(True)
        self._btn_calib.clicked.connect(self._cmd_calib_zero)
        laser_lay.addWidget(self._btn_calib, 1)
        right_col.addWidget(laser_grp)
        
        # 日志占位(调试模式时log_group移到这里)
        self._debug_log_slot = QVBoxLayout()
        right_col.addLayout(self._debug_log_slot, 1)
        two_col.addLayout(right_col)
        
        return panel


    def _set_step_size(self, val):
        """设置步进量并更新按钮选中状态"""
        self._step_size = float(val)
        for btn, v in self._step_btns:
            btn.setChecked(v == val)
        self.log(f"⚙ 步进量设为 {val}°")

    def _cmd_xy_move(self, dx_sign, dy_sign):
        """XY方向移动(框架stub)
        Args:
            dx_sign: -1/0/1 表示X方向
            dy_sign: -1/0/1 表示Y方向
        TODO: 实际发送 0x01 帧(绝对坐标) 到STM32
        """
        dx = dx_sign * self._step_size  # 单位°
        dy = dy_sign * self._step_size
        self._current_x = max(0.0, min(2475.0, self._current_x + dx))
        self._current_y = max(0.0, min(2475.0, self._current_y + dy))
        self._update_coord_display()
        self._set_motion_state("moving")
        self.log(f"▸ XY移动 → X={self._current_x:.1f} Y={self._current_y:.1f}°")
        self._send_cmd(0x01, self._current_x, self._current_y)
        self._homed = False   # 移动后已离开零点, 需重新回零才能再校准

    def _cmd_z_move(self, direction):
        """Z轴步进移动(框架stub)
        Args:
            direction: 1=上, -1=下
        TODO: 实际发送 0x06 帧(Z步数) 到STM32
        """
        # direction: 1=上(Z减小), -1=下(Z增大). Z向下为正方向, 范围[0,950]
        delta = int(-direction * self._step_size)  # 下=正
        target = max(0, min(950, self._current_z + delta))
        actual = target - self._current_z
        if actual == 0:
            self.log("⚠ Z轴已到限位")
            return
        self._current_z = target
        self._update_coord_display()
        self._set_motion_state("moving")
        self.log(f"▸ Z轴{'下' if actual>0 else '上'} {abs(actual)}° (当前{target}°)")
        self._send_cmd(0x06, actual)
        self._homed = False   # 移动后已离开零点, 需重新回零才能再校准

    def _popup_numpad(self, line_edit, title):
        """点击输入框弹出自定义数字键盘"""
        dlg = NumPadDialog(self, title=title, init_value=line_edit.text())
        # 居中显示在主窗口
        dlg.move(self.geometry().center().x() - dlg.width()//2,
                 self.geometry().center().y() - dlg.height()//2)
        if dlg.exec_() == QDialog.Accepted:
            line_edit.setText(dlg.get_value())

    def _cmd_goto_xy(self):
        """运动到指定XYZ坐标: 三轴联动(单帧0x08), 同时启动避免分帧被门控拒绝。
        XY为绝对坐标(0.1mm), Z发送相对当前的步进增量(与move_z步数语义一致)。
        """
        try:
            tx = max(0.0, min(2475.0, float(self._input_x.text())))
            ty = max(0.0, min(2475.0, float(self._input_y.text())))
            tz = max(0.0, min(950.0, float(self._input_z.text())))
            dz = tz - self._current_z
            self._current_x = tx
            self._current_y = ty
            self._current_z = tz
            self._update_coord_display()
            self._set_motion_state("moving")
            self.log(f"▸ 运动到 X={tx:.1f} Y={ty:.1f} Z={tz:.1f}°")
            self._send_cmd(0x08, tx, ty, int(dz))
            self._homed = False   # 移动后已离开零点, 需重新回零才能再校准
        except ValueError:
            self.log("⚠ 坐标输入无效")

    def _cmd_home(self):
        """回零(框架stub)
        TODO: 发送0x03帧
        """
        self._current_x = 0.0
        self._current_y = 0.0
        self._current_z = 0
        self._update_coord_display()
        self._set_motion_state("moving")
        self.log("▸ 回零 (XY+Z)")
        self._send_cmd(0x03)
        # 回零后状态干净, 允许绝对零点校准(防止未回零误触乱套)
        self._homed = True

    def _cmd_calib_zero(self):
        """绝对零点校准: 将当前物理位置标定为工作坐标(-360,-360)。
        前提: 已手动将电机调到绝对零点并回过零。坐标由STM32重算后上报刷新。
        防误触: 单次上电仅允许校准一次。
        """
        if getattr(self, '_calibrated', False):
            self.log("⚠ 本次上电已校准过, 如需重新校准请重启系统")
            return
        if not getattr(self, '_homed', False):
            self.log("⚠ 请先回零再执行绝对零点校准")
            return
        self.log("▸ 绝对零点校准 → 标定当前位置为 (-254.56, -254.56)")
        self._send_cmd(0x09)
        self._calibrated = True

    def _cmd_estop(self):
        """急停(框架stub)
        TODO: 发送0x02帧
        """
        self._set_motion_state("estop")
        self.log("🛑 急停")
        self._send_cmd(0x02)

    def _cmd_squeeze(self):
        """挤锡测试(框架stub)
        TODO: 发送0x07帧(挤锡1次)
        """
        self.log("💧 挤锡1次")
        self._send_cmd(0x07, 1)

    def _on_conf_changed(self, val):
        """置信度spin变化时，更新推理线程的阈值(实时生效)"""
        if self.infer_thread is not None:
            self.infer_thread.conf_thresh = float(val)

    def _poll_motor_state(self):
        """主线程轮询motor真实坐标/状态(线程安全),更新调试显示。"""
        if not self._motor:
            return
        try:
            x, y = self._motor.get_position()
            z = self._motor.get_z() if hasattr(self._motor, 'get_z') else self._current_z
            st = self._motor.get_state()
        except Exception:
            return
        # 同步逻辑坐标(以STM32上报为准)
        self._current_x, self._current_y, self._current_z = x, y, z
        if self.current_mode == "debug":
            self._update_coord_display()
            if hasattr(self, '_lbl_laser'):
                try:
                    self._lbl_laser.setText(f"距离: {self._motor.get_laser()} mm")
                except Exception:
                    pass
            self._set_motion_state({0: "idle", 1: "moving", 2: "estop"}.get(st, "idle"))

    def _update_coord_display(self):
        """更新调试面板的XYZ坐标显示"""
        if hasattr(self, '_lbl_coord_x'):
            self._lbl_coord_x.setText(f"X: {self._current_x:.1f}")
            self._lbl_coord_y.setText(f"Y: {self._current_y:.1f}")
            if hasattr(self, '_lbl_coord_z'):
                self._lbl_coord_z.setText(f"Z: {self._current_z:.1f}")

    def _set_motion_state(self, state):
        """更新运动状态指示灯
        Args:
            state: 'idle'|'moving'|'estop'
        """
        if not hasattr(self, '_dbg_status_led'):
            return
        if state == "moving":
            self._dbg_status_led.setText("🟢")
            self._dbg_status_txt.setText("运动中")
        elif state == "estop":
            self._dbg_status_led.setText("🔴")
            self._dbg_status_txt.setText("急停")
        else:
            self._dbg_status_led.setText("⚪")
            self._dbg_status_txt.setText("已停止")

    def _send_cmd(self, cmd_id, *args):
        """[接口预留] 发送命令到STM32
        协议: AA 55 ID [payload] checksum
        当前为stub，待STM32端USB CDC命令解析完成后实现
        """
        # 通过motor_control发送(已封装,见motor_control.py)
        if self._motor is None:
            self.log(f"⚠ 运动控制器未连接 (id=0x{cmd_id:02X})")
            return False
        # cmd_id分派到motor接口
        try:
            if cmd_id == 0x01:
                return bool(self._motor.move_to(args[0], args[1]))
            elif cmd_id == 0x02:
                return bool(self._motor.estop())
            elif cmd_id == 0x03:
                return bool(self._motor.home())
            elif cmd_id == 0x06:
                return bool(self._motor.move_z(args[0]))
            elif cmd_id == 0x07:
                return bool(self._motor.squeeze(args[0] if args else 1))
            elif cmd_id == 0x08:
                if hasattr(self._motor, 'move_xyz'):
                    return bool(self._motor.move_xyz(args[0], args[1], args[2]))
                self.log("⚠ 当前motor_control不支持XYZ联动(0x08)")
                return False
            elif cmd_id == 0x09:
                if hasattr(self._motor, 'calibrate_zero'):
                    return bool(self._motor.calibrate_zero())
                self.log("⚠ 当前motor_control不支持绝对零点校准(0x09)")
                return False
            else:
                self.log(f"⚠ 未知指令 id=0x{cmd_id:02X}")
                return False
        except Exception as e:
            self.log(f"✗ 指令发送出错: {e}")
            return False





    def closeEvent(self, event):
        """窗口关闭事件：停止推理线程，释放资源"""
        try:
            if hasattr(self, '_health_thread') and self._health_thread:
                self._health_thread.stop()
        except Exception:
            pass
        self._stop_tip_preview()
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
