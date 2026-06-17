#!/usr/bin/env python3
"""
数据集拍摄工具 (独立脚本)
========================
功能: 摄像头1920x1080采集，显示中心1080x1080区域，支持双指缩放，
      保存放大后可见区域为数据集图片(用于YOLOv5训练)。

操作:
    - 双指缩放/拖动选择感兴趣区域
    - 点击拍摄按钮保存当前视野(放大后的裁剪图)
    - 图片保存到SD卡 /run/media/mmcblk1p1/datasheet/

硬件: RK3588 ELF2开发板 + USB摄像头(/dev/video21)
"""
import sys
import os
import cv2
import numpy as np
import time
import glob

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPainterPath
from PyQt5.QtCore import QRectF


# ===== 配置 =====
CAMERA_ID = 21
SAVE_DIRS = [
    '/media/elf/OPI_BOOT/datasheet',  # SD卡优先
    '/run/media/mmcblk1p1/datasheet',  # SD卡备选
    '/home/elf/solder_system/datasheet',  # 回退路径
]

def get_save_dir():
    """获取保存目录，优先SD卡，无权限则回退"""
    for d in SAVE_DIRS:
        try:
            os.makedirs(d, exist_ok=True)
            test_f = os.path.join(d, '.write_test')
            with open(test_f, 'w') as f:
                f.write('t')
            os.remove(test_f)
            return d
        except Exception:
            continue
    return '/tmp/datasheet'


def get_scale():
    """自适应缩放系数，基准1024"""
    screen = QApplication.primaryScreen().geometry()
    return screen.width() / 1024


class PinchZoomLabel(QLabel):
    """支持双指缩放的图像显示(与主系统相同实现)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grabGesture(Qt.PinchGesture)
        self._zoom = 1.0
        self._min_zoom = 1.0
        self._max_zoom = 8.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._original_pixmap = None
        self._current_display = None
        self._gesture_in_progress = False

    def setDisplayPixmap(self, pixmap):
        self._original_pixmap = pixmap
        self._apply_transform()

    def reset_zoom(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._apply_transform()

    def get_crop_rect(self):
        """返回当前视野对应原图的裁剪区域(x, y, w, h)比例"""
        if not self._original_pixmap or self._zoom <= 1.0:
            return (0, 0, 1.0, 1.0)
        pm = self._original_pixmap
        label_w, label_h = self.width(), self.height()
        base_scale = min(label_w / pm.width(), label_h / pm.height())
        w = int(pm.width() * base_scale * self._zoom)
        h = int(pm.height() * base_scale * self._zoom)
        cx = w / 2 - self._pan_x
        cy = h / 2 - self._pan_y
        crop_w = min(label_w, w)
        crop_h = min(label_h, h)
        x = int(max(0, min(cx - crop_w / 2, w - crop_w)))
        y = int(max(0, min(cy - crop_h / 2, h - crop_h)))
        # 映射回原图比例
        rx = x / w
        ry = y / h
        rw = crop_w / w
        rh = crop_h / h
        return (rx, ry, rw, rh)

    def paintEvent(self, event):
        if self._current_display:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            path = QPainterPath()
            s = get_scale()
            path.addRoundedRect(QRectF(self.rect()), 14*s, 14*s)
            painter.setClipPath(path)
            pm = self._current_display
            x = (self.width() - pm.width()) // 2
            y = (self.height() - pm.height()) // 2
            painter.drawPixmap(x, y, pm)
            painter.end()
        else:
            super().paintEvent(event)

    def _apply_transform(self):
        if not self._original_pixmap:
            return
        pm = self._original_pixmap
        label_w = self.width() or pm.width()
        label_h = self.height() or pm.height()
        transform_mode = Qt.SmoothTransformation

        if self._zoom <= 1.0:
            scaled = pm.scaled(label_w, label_h, Qt.KeepAspectRatio, transform_mode)
            self._current_display = scaled
            self.update()
            self._pan_x = 0.0
            self._pan_y = 0.0
            return

        base_scale = min(label_w / pm.width(), label_h / pm.height())
        w = int(pm.width() * base_scale * self._zoom)
        h = int(pm.height() * base_scale * self._zoom)
        scaled = pm.scaled(w, h, Qt.KeepAspectRatio, transform_mode)
        cx = w / 2 - self._pan_x
        cy = h / 2 - self._pan_y
        crop_w = min(label_w, w)
        crop_h = min(label_h, h)
        x = int(max(0, min(cx - crop_w / 2, w - crop_w)))
        y = int(max(0, min(cy - crop_h / 2, h - crop_h)))
        cropped = scaled.copy(x, y, crop_w, crop_h)
        self._current_display = cropped
        self.update()

    def event(self, ev):
        if ev.type() == ev.Gesture:
            return self._gesture_event(ev)
        return super().event(ev)

    def _gesture_event(self, ev):
        pinch = ev.gesture(Qt.PinchGesture)
        if pinch:
            self._gesture_in_progress = True
            factor = pinch.scaleFactor()
            self._zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * factor))
            delta = pinch.centerPoint() - pinch.lastCenterPoint()
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._apply_transform()
            if pinch.state() == pinch.GestureFinished:
                self._gesture_in_progress = False
        return True


class CaptureWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据集拍摄工具")
        screen = QApplication.primaryScreen().geometry()
        self.setFixedSize(screen.width(), screen.height())
        self.showFullScreen()

        s = get_scale()
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(int(8*s), int(8*s), int(8*s), int(8*s))

        # 图像显示
        self.video_label = PinchZoomLabel("正在启动摄像头...")
        self.video_label.setAlignment(Qt.AlignCenter)
        # 强制正方形显示区域(边长=屏幕高度-按钮栏-边距)
        screen_h = QApplication.primaryScreen().geometry().height()
        square_size = screen_h - int(80 * s)
        self.video_label.setFixedSize(square_size, square_size)
        self.video_label.setStyleSheet(
            f"background: #1c1c1e; color: #8e8e93; border-radius: {int(14*s)}px; font-size: {int(14*s)}px;"
        )
        layout.addWidget(self.video_label, 1)

        # 底部按钮栏
        btn_layout = QHBoxLayout()
        self.lbl_info = QLabel("就绪")
        self.lbl_info.setStyleSheet(f"font-size: {int(12*s)}px; color: #8e8e93;")
        btn_layout.addWidget(self.lbl_info)
        btn_layout.addStretch()

        self.btn_capture = QPushButton(f"📷 拍摄")
        self.btn_capture.setFixedHeight(int(48*s))
        self.btn_capture.setFixedWidth(int(120*s))
        self.btn_capture.setStyleSheet(f"""
            QPushButton {{
                background: #007aff; color: white; border: none;
                border-radius: {int(10*s)}px; font-size: {int(14*s)}px; font-weight: 600;
            }}
            QPushButton:pressed {{ background: #0056b3; }}
        """)
        self.btn_capture.clicked.connect(self.capture)
        btn_layout.addWidget(self.btn_capture)

        btn_layout.addSpacing(int(10*s))
        self.btn_quit = QPushButton("✕ 退出")
        self.btn_quit.setFixedHeight(int(48*s))
        self.btn_quit.setFixedWidth(int(100*s))
        self.btn_quit.setStyleSheet(f"""
            QPushButton {{
                background: #ff3b30; color: white; border: none;
                border-radius: {int(10*s)}px; font-size: {int(14*s)}px; font-weight: 600;
            }}
        """)
        self.btn_quit.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_quit)

        layout.addLayout(btn_layout)

        # 摄像头
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.current_frame = None
        self.save_dir = get_save_dir()
        self.count = self._get_next_idx()

        # 定时刷新
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30fps

        self.lbl_info.setText(f"保存到: {self.save_dir} | 已有 {self.count} 张")

    def _get_next_idx(self):
        """获取下一张图片编号"""
        existing = glob.glob(os.path.join(self.save_dir, '*.jpg'))
        if not existing:
            return 0
        nums = []
        for f in existing:
            base = os.path.splitext(os.path.basename(f))[0]
            try:
                nums.append(int(base))
            except ValueError:
                pass
        return max(nums) + 1 if nums else 0

    def update_frame(self):
        """读帧，裁中心1080x1080显示"""
        ret, frame = self.cap.read()
        if not ret:
            return
        h, w = frame.shape[:2]
        # 裁剪中心1080x1080
        crop_size = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = cx - crop_size // 2
        y1 = cy - crop_size // 2
        self.current_frame = frame[y1:y1+crop_size, x1:x1+crop_size]

        # 转QPixmap显示
        rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self.video_label.setDisplayPixmap(QPixmap.fromImage(qimg))


    def capture(self):
        """拍摄：保存当前放大后视野，resize到1080x1080（所见即所得）"""
        if self.current_frame is None:
            return

        # 获取当前视野裁剪比例(相对于1080x1080原图)
        rx, ry, rw, rh = self.video_label.get_crop_rect()
        h, w = self.current_frame.shape[:2]  # 1080x1080
        x1 = int(rx * w)
        y1 = int(ry * h)
        x2 = x1 + int(rw * w)
        y2 = y1 + int(rh * h)
        # 裁剪当前视野
        crop = self.current_frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = self.current_frame
        # 确保裁剪为正方形后再resize到1080x1080
        ch, cw = crop.shape[:2]
        if cw != ch:
            side = min(cw, ch)
            cx, cy = cw // 2, ch // 2
            crop = crop[cy-side//2:cy+side//2, cx-side//2:cx+side//2]
        save_img = cv2.resize(crop, (1080, 1080), interpolation=cv2.INTER_LINEAR)

        # 保存
        filename = f"{self.count:04d}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, save_img)
        self.count += 1
        self.lbl_info.setText(f"✓ 已保存 {filename} (1080x1080) | 共 {self.count} 张")


    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    s = get_scale()
    app.setFont(QFont("PingFang SC", max(9, int(9 * s))))
    win = CaptureWindow()
    win.show()
    sys.exit(app.exec_())
