#!/usr/bin/env python3
"""
数据集采集工具
=============
功能: 从USB摄像头实时预览画面，按键采集训练样本图片。
用途: 为YOLOv5焊盘/缺陷检测模型收集训练数据。

操作:
    Enter - 保存当前帧为 XXXX.jpg (自动递增编号)
    q     - 退出采集

输出目录: /home/elf/solder_system/datasheet/
图片格式: BGR JPG, 分辨率取决于摄像头默认设置
"""
import cv2
import os

SAVE_DIR = "/home/elf/solder_system/datasheet"
CAM_ID = 21

os.makedirs(SAVE_DIR, exist_ok=True)

# 找到当前最大编号
existing = [f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg') and f[:-4].isdigit()]
if existing:
    start_idx = max(int(f[:-4]) for f in existing) + 1
else:
    start_idx = 1

cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    print(f"无法打开摄像头 /dev/video{CAM_ID}")
    exit(1)

print(f"数据集采集工具")
print(f"保存目录: {SAVE_DIR}")
print(f"起始编号: {start_idx}")
print(f"操作: Enter=拍照  q=退出")
print("-" * 40)

idx = start_idx
while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败")
        continue

    cv2.imshow("采集预览 (Enter=拍照, q=退出)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 13 or key == 10:  # Enter
        filename = f"{idx:04d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"✓ 已保存: {filename} ({frame.shape[1]}x{frame.shape[0]})")
        idx += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n采集完成，共 {idx - start_idx} 张，保存在 {SAVE_DIR}")
