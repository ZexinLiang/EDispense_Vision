#!/usr/bin/env python3
"""
执行系统(STM32)心跳检测脚本
============================
功能: 通过USB CDC串口向STM32发送心跳信号，检测执行系统是否在线。
调用方式: 由UI主程序每10秒通过subprocess调用。
退出码: 0 = 在线(收到正确回显), 1 = 离线(设备不存在/无响应/回显错误)

通信协议:
    发送: "HEARTBEAT\n"
    期望回显: 包含 "HEARTBEAT" 的数据(STM32固件设置为全回显)
"""
import serial
import glob
import sys
import time

# 自动查找可用的USB CDC设备(支持热插拔后设备号变化)
ports = sorted(glob.glob("/dev/ttyACM*"))
if not ports:
    sys.exit(1)  # 无设备 → 离线

try:
    # 短连接模式: 打开 → 通信 → 关闭，避免长连接占用导致设备拔出检测失败
    ser = serial.Serial(ports[0], 115200, timeout=1)
    time.sleep(0.1)           # 等待串口稳定
    ser.reset_input_buffer()  # 清空可能残留的旧数据
    ser.write(b"HEARTBEAT\n")
    ser.flush()               # 确保数据立刻发出
    time.sleep(0.4)           # 等待STM32处理并回显
    resp = ser.read(256)      # 读取回显数据
    ser.close()
    if b"HEARTBEAT" in resp:
        sys.exit(0)  # 收到正确回显 → 在线
    else:
        sys.exit(1)  # 回显内容异常 → 离线
except Exception:
    sys.exit(1)  # 任何异常(设备不可用/通信失败) → 离线
