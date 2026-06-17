#!/usr/bin/env python3
"""
运动控制模块 (Motor Control)
============================
封装 RK3588 上位机 → STM32 执行机构的串口通信。
协议: TJC帧格式  AA 55 | CMD_ID | LEN | PAYLOAD | CHECKSUM
传输: USB CDC 虚拟串口 /dev/ttyACM0, 115200

命令ID (与STM32端 CDC_Receive_FS 解析对应):
    0x01  XY绝对坐标移动   payload: int16 x, int16 y (单位0.1mm, 大端)
    0x02  急停            payload: 无
    0x03  回零            payload: 无
    0x06  Z轴步进         payload: int16 steps (正=上, 负=下)
    0x07  挤锡            payload: uint8 count

用法:
    from motor_control import MotorController
    mc = MotorController(port="/dev/ttyACM0")
    mc.connect()
    mc.move_to(100.0, 50.0)   # 移动到 X=100mm Y=50mm
    mc.home()
    pos = mc.get_position()   # (x, y) 上位机维护的逻辑坐标
    mc.close()

注意:
    - 坐标范围软限位 0~247.5mm (对应STM32端 2475 的 0.1° 角度限制)
    - 当前STM32端USB CDC仅回显, 命令解析待实现 (见文件末尾协议说明)
    - get_position() 返回上位机维护的逻辑坐标; 如需真实坐标需STM32主动上报(0x10帧,待扩展)
"""

import struct
import time
import glob

try:
    import serial
except ImportError:
    serial = None


# ===== 协议常量 =====
FRAME_HEAD = b'\xAA\x55'
CMD_MOVE_XY   = 0x01   # XY绝对坐标
CMD_ESTOP     = 0x02   # 急停
CMD_HOME      = 0x03   # 回零
CMD_MOVE_Z    = 0x06   # Z轴步进
CMD_SQUEEZE   = 0x07   # 挤锡

# ===== 机械参数 =====
AXIS_MIN_MM = 0.0
AXIS_MAX_MM = 247.5     # 软限位 (STM32端 2475 * 0.1°)
MM_TO_UNIT  = 10        # 1mm = 10 个 0.1mm 单位


def _build_frame(cmd_id, payload=b''):
    """组装TJC帧: AA 55 | CMD | LEN | PAYLOAD | CHECKSUM(异或)

    Args:
        cmd_id: 命令字节
        payload: bytes 负载
    Returns:
        完整帧 bytes
    """
    length = len(payload)
    body = bytes([cmd_id, length]) + payload
    checksum = 0
    for b in body:
        checksum ^= b
    return FRAME_HEAD + body + bytes([checksum])


class MotorController:
    """执行机构串口控制器。线程不安全，单线程调用或自行加锁。"""

    def __init__(self, port=None, baudrate=115200, timeout=1.0, logger=None):
        """
        Args:
            port: 串口设备路径; None则自动查找 /dev/ttyACM*
            baudrate: 波特率
            timeout: 读超时(秒)
            logger: 可选回调 logger(msg:str), 默认print
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self._log = logger or (lambda m: print(f"[motor] {m}"))
        # 上位机维护的逻辑坐标 (mm)
        self._x = 0.0
        self._y = 0.0

    # ---------- 连接管理 ----------
    def connect(self):
        """打开串口。自动查找设备。返回True成功。"""
        if serial is None:
            self._log("pyserial未安装")
            return False
        port = self.port
        if port is None:
            ports = sorted(glob.glob('/dev/ttyACM*'))
            if not ports:
                self._log("未找到 /dev/ttyACM* 设备")
                return False
            port = ports[0]
        try:
            self.ser = serial.Serial(port, self.baudrate, timeout=self.timeout)
            time.sleep(0.1)
            self.ser.reset_input_buffer()
            self.port = port
            self._log(f"已连接 {port} @ {self.baudrate}")
            return True
        except Exception as e:
            self._log(f"连接失败: {e}")
            self.ser = None
            return False

    def is_connected(self):
        """返回串口是否已连接"""
        return self.ser is not None and self.ser.is_open

    def close(self):
        """关闭串口连接"""
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    # ---------- 底层发送 ----------
    def _send(self, cmd_id, payload=b''):
        """发送一帧。返回True成功。"""
        frame = _build_frame(cmd_id, payload)
        if not self.is_connected():
            self._log(f"未连接, 丢弃帧 id=0x{cmd_id:02X} {frame.hex()}")
            return False
        try:
            self.ser.write(frame)
            self.ser.flush()
            self._log(f"发送 id=0x{cmd_id:02X} payload={payload.hex()} frame={frame.hex()}")
            return True
        except Exception as e:
            self._log(f"发送失败: {e}")
            return False

    @staticmethod
    def _clamp(v):
        """将坐标值限制在软限位范围 [0, 247.5] mm 内"""
        return max(AXIS_MIN_MM, min(AXIS_MAX_MM, v))

    # ---------- 高层接口 (供UI/脚本调用) ----------
    def move_to(self, x_mm, y_mm):
        """移动到绝对坐标 (mm)。自动软限位。"""
        x = self._clamp(x_mm)
        y = self._clamp(y_mm)
        payload = struct.pack('>hh', int(x * MM_TO_UNIT), int(y * MM_TO_UNIT))
        ok = self._send(CMD_MOVE_XY, payload)
        if ok:
            self._x, self._y = x, y
        return ok

    def move_xy_rel(self, dx_mm, dy_mm):
        """相对当前坐标移动 (mm)。内部转为绝对坐标。"""
        return self.move_to(self._x + dx_mm, self._y + dy_mm)

    def move_z(self, steps):
        """Z轴步进。steps正=上, 负=下 (步数, 非mm)。"""
        payload = struct.pack('>h', int(steps))
        return self._send(CMD_MOVE_Z, payload)

    def home(self):
        """回零。回零后逻辑坐标归零。"""
        ok = self._send(CMD_HOME)
        if ok:
            self._x, self._y = 0.0, 0.0
        return ok

    def estop(self):
        """急停。"""
        return self._send(CMD_ESTOP)

    def squeeze(self, count=1):
        """挤锡 count 次。"""
        payload = struct.pack('>B', int(count) & 0xFF)
        return self._send(CMD_SQUEEZE, payload)

    def get_position(self):
        """返回上位机维护的逻辑坐标 (x_mm, y_mm)。"""
        return (self._x, self._y)

    def set_position(self, x_mm, y_mm):
        """手动设置逻辑坐标(不发送指令), 用于同步真实位置。"""
        self._x, self._y = x_mm, y_mm


# ===== 自测 =====
if __name__ == '__main__':
    import sys
    mc = MotorController()
    if not mc.connect():
        print("连接失败，退出")
        sys.exit(1)
    print("帧编码自测:")
    print("  move_to(100,50):", _build_frame(CMD_MOVE_XY, struct.pack('>hh', 1000, 500)).hex())
    print("  home:", _build_frame(CMD_HOME).hex())
    print("  estop:", _build_frame(CMD_ESTOP).hex())
    print("  move_z(-50):", _build_frame(CMD_MOVE_Z, struct.pack('>h', -50)).hex())
    print("  squeeze(1):", _build_frame(CMD_SQUEEZE, struct.pack('>B', 1)).hex())
    mc.close()


# ============================================================
# STM32端待实现 (CDC_Receive_FS 解析参考):
#   解析帧: AA 55 | CMD | LEN | PAYLOAD | CHECKSUM(异或CMD..PAYLOAD)
#   case 0x01: int16 x = (payload[0]<<8)|payload[1]; y同理; move_axis_to(x/10.0, y/10.0)
#   case 0x02: g_SystemState=EMERGENCY_STOP; Stepper_Stop(all)
#   case 0x03: move_axis_to(0,0)
#   case 0x06: int16 steps; 直接步进 stepper3
#   case 0x07: squeeze_paste(payload[0])
# ============================================================
