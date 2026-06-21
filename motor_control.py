#!/usr/bin/env python3
"""
运动控制模块 (Motor Control) - 安全门控版
==========================================
封装 RK3588 上位机 → STM32 执行机构的串口通信, 带运动状态门控。

下行协议 (TJC帧, 与STM32 TJC_RxCallback一致):
    AA 55 | CMD | PAYLOAD | CHECKSUM(=cmd+payload各字节累加和 & 0xFF)
    0x01 XY绝对移动  payload: int16 x, int16 y (大端, 0.1mm)   [STM32已实现]
    0x02 急停        payload: 无                                [STM32已实现]
    0x03 回零        payload: 无                                [STM32已实现]
    0x06 Z轴步进     payload: int16 steps (正上负下)            [STM32待加]
    0x07 挤锡        payload: uint8 count                       [STM32待加]
    0x08 XYZ联动移动 payload: int16 x, int16 y, int16 z (大端)   [STM32待加]
                     一帧三轴同时启动, 避免分两帧发被门控拒绝
    0x09 绝对零点校准 payload: 无                                [STM32待加]
                     将当前物理位置标定为工作坐标(-360,-360), 不移动电机

上行协议 (STM32 → RK3588, 待STM32实现):
    AA 55 | CMD | PAYLOAD | CHECKSUM
    0x10 状态上报(周期~100ms) payload: int16 x, int16 y, int16 z, uint8 state
                              state: 0=空闲 1=运动中 2=急停
    0x11 动作完成            payload: uint8 cmd_id (完成的命令)

安全门控逻辑:
    - 运动命令(move_to/move_xy_rel/move_z/move_xyz/home/squeeze)发出后置 busy=True
    - busy期间拒绝新运动命令(返回False+告警), 防止指令堆叠撞坏机构
    - 急停 estop() 任何时候都可发送(安全优先), 发送后立即清 busy
    - busy 由上行帧解除: 收到 0x11完成 或 0x10状态=空闲 → busy=False
    - 容错: 若STM32未实现上报(从未收到上行帧), 启用超时保护
            (BUSY_TIMEOUT_S 后自动解除, 避免永久卡死)
"""

import struct
import time
import glob
import threading

try:
    import serial
except ImportError:
    serial = None

# ===== 协议常量 =====
FRAME_HEAD = b'\xAA\x55'
CMD_MOVE_XY = 0x01
CMD_ESTOP   = 0x02
CMD_HOME    = 0x03
CMD_MOVE_Z  = 0x06
CMD_SQUEEZE = 0x07
CMD_MOVE_XYZ = 0x08
CMD_CALIB_ZERO = 0x09
# 上行
REP_STATUS  = 0x10
REP_DONE    = 0x11

# ===== 机械参数 =====
AXIS_MIN_MM = 0.0
AXIS_MAX_MM = 2475.0
MM_TO_UNIT  = 10

# ===== 状态 =====
STATE_IDLE = 0
STATE_MOVING = 1
STATE_ESTOP = 2

# 若STM32未实现上报, busy超时自动解除(秒)。设为运动最坏耗时余量。
BUSY_TIMEOUT_S = 8.0


def _build_frame(cmd_id, payload=b''):
    """组装TJC帧: AA55|CMD|PAYLOAD|CHECKSUM(累加和&0xFF)"""
    body = bytes([cmd_id]) + payload
    return FRAME_HEAD + body + bytes([sum(body) & 0xFF])


class MotorController:
    """执行机构串口控制器, 带运动状态门控。"""

    def __init__(self, port=None, baudrate=115200, timeout=1.0, logger=None):
        """初始化: 端口/波特率/状态变量/回调; port=None时自动查找ttyACM*"""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self._log = logger or (lambda m: print(f"[motor] {m}"))
        self._x = 0.0
        self._y = 0.0
        self._z = 0
        self._laser = 0
        # 状态门控
        self._state = STATE_IDLE
        self._busy = False
        self._busy_since = 0.0
        self._got_report = False   # 是否收到过上行帧
        self._lock = threading.Lock()
        # 上行回调
        self._on_status = None     # callback(x_mm, y_mm, z_raw, state)
        self._on_done = None       # callback(cmd_id)
        # 接收线程
        self._rx_thread = None
        self._rx_running = False
        self._last_rx_time = 0.0   # 最后收到上行帧的时间(供is_online判定)

    # ---------- 回调注册 ----------
    def set_callbacks(self, on_status=None, on_done=None):
        """注册上行回调: on_status(x,y,z,state), on_done(cmd_id)。"""
        if on_status:
            self._on_status = on_status
        if on_done:
            self._on_done = on_done

    # ---------- 连接管理 ----------
    def connect(self):
        """打开串口并启动接收线程。"""
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
            self.ser = serial.Serial(port, self.baudrate, timeout=0.2)
            time.sleep(0.1)
            self.ser.reset_input_buffer()
            self.port = port
            self._log(f"✓ 已连接 {port} @ {self.baudrate}")
            self._start_rx()
            return True
        except Exception as e:
            self._log(f"连接失败: {e}")
            self.ser = None
            return False

    def is_connected(self):
        """返回串口是否已连接"""
        return self.ser is not None and self.ser.is_open

    def close(self):
        """停止接收线程并关闭串口"""
        self._stop_rx()
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    # ---------- 接收线程 ----------
    def _start_rx(self):
        """启动后台线程接收并解析STM32上行帧"""
        self._rx_running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def _stop_rx(self):
        """停止上行接收线程"""
        self._rx_running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None

    def _rx_loop(self):
        """接收线程: 解析上行帧 AA55|CMD|PAYLOAD|SUM。"""
        buf = bytearray()
        # 各上行命令的payload长度
        plen = {REP_STATUS: 9, REP_DONE: 1}
        while self._rx_running:
            try:
                data = self.ser.read(64)
            except Exception:
                # 设备断开(拔USB), 尝试重连而非退出线程
                self._reconnect()
                buf = bytearray()
                continue
            if not data:
                continue
            buf.extend(data)
            # 解析: 找帧头AA55
            while len(buf) >= 3:
                if buf[0] != 0xAA:
                    buf.pop(0); continue
                if buf[1] != 0x55:
                    buf.pop(0); continue
                cmd = buf[2]
                if cmd not in plen:
                    # 未知命令, 丢弃帧头继续找
                    buf.pop(0); continue
                need = 3 + plen[cmd] + 1   # head(2)+cmd(1)+payload+sum(1)
                if len(buf) < need:
                    break   # 等更多数据
                frame = bytes(buf[:need])
                body = frame[2:-1]   # cmd+payload
                if (sum(body) & 0xFF) == frame[-1]:
                    self._handle_report(cmd, frame[3:-1])
                del buf[:need]


    def _reconnect(self):
        """串口断开后尝试重连(拔插USB恢复)。阻塞重试直到成功或线程停止。"""
        try:
            if self.ser: self.ser.close()
        except Exception:
            pass
        self.ser = None
        while self._rx_running:
            time.sleep(0.3)
            try:
                import glob as _g
                ports = sorted(_g.glob('/dev/ttyACM*'))
                if not ports:
                    continue
                self.ser = serial.Serial(ports[0], self.baudrate, timeout=0.2)
                self.port = ports[0]
                self.ser.reset_input_buffer()
                self._log(f"✓ 串口重连成功 {ports[0]}")
                return
            except Exception:
                self.ser = None

    def _handle_report(self, cmd, payload):
        """处理上行帧, 更新状态并回调。"""
        self._got_report = True
        self._last_rx_time = time.time()   # 记录最后收到上行帧的时间(供is_online判定)
        if cmd == REP_STATUS and len(payload) == 9:
            ix, iy, iz, ilaser = struct.unpack('>hhhh', payload[:8])
            state = payload[8]
            with self._lock:
                self._x = ix / 10.0
                self._y = iy / 10.0
                self._z = iz / 10.0
                self._laser = ilaser
                self._state = state
                self._busy = (state == STATE_MOVING)
            if self._on_status:
                try:
                    self._on_status(self._x, self._y, self._z, state)
                except Exception:
                    pass
        elif cmd == REP_DONE and len(payload) == 1:
            with self._lock:
                self._busy = False
                if self._state == STATE_MOVING:
                    self._state = STATE_IDLE
            if self._on_done:
                try:
                    self._on_done(payload[0])
                except Exception:
                    pass

    # ---------- busy门控 ----------
    def is_online(self):
        """执行系统是否在线: 最近3秒内收到过STM32上行帧(0x10/0x11)。
        STM32固件每100ms主动发0x10状态帧, 故3秒窗口足够判定。"""
        return (time.time() - self._last_rx_time) < 3.0

    def is_busy(self):
        """是否运动中。带超时保护: 未收到上报且超时则自动解除。"""
        with self._lock:
            if self._busy and not self._got_report:
                if time.time() - self._busy_since > BUSY_TIMEOUT_S:
                    self._busy = False
                    self._state = STATE_IDLE
            return self._busy

    def get_laser(self):
        """返回最近上报的激光测距值(mm)。"""
        with self._lock:
            return self._laser

    def get_z(self):
        """返回最近上报的Z轴角度位置(°)。"""
        with self._lock:
            return self._z

    def get_state(self):
        """返回当前状态码: 0空闲/1运动/2急停。"""
        with self._lock:
            return self._state

    def _set_busy(self):
        """标记进入运动忙状态, 记录起始时间用于超时保护"""
        with self._lock:
            self._busy = True
            self._busy_since = time.time()
            if self._state != STATE_ESTOP:
                self._state = STATE_MOVING

    # ---------- 底层发送 ----------
    def _send(self, cmd_id, payload=b''):
        """组帧并写入串口, 记录日志, 返回是否成功"""
        frame = _build_frame(cmd_id, payload)
        if not self.is_connected():
            self._log(f"⚠ 未连接,丢弃「0x{cmd_id:02X}」{frame.hex()}")
            return False
        try:
            self.ser.write(frame)
            self.ser.flush()
            self._log(f"「TX 0x{cmd_id:02X}」{frame.hex()}")
            # 原始TX字节落盘, 方便SSH实时监看: tail -f /tmp/motor_tx.log
            try:
                import time as _t
                with open('/tmp/motor_tx.log', 'a') as _f:
                    _f.write(f"{_t.strftime('%H:%M:%S')} TX {frame.hex()}\n")
            except Exception:
                pass
            return True
        except Exception as e:
            self._log(f"✗ 发送失败: {e}")
            return False

    @staticmethod
    def _clamp(v):
        """将坐标限制在软限位 [0, 2475] 单位°内"""
        return max(AXIS_MIN_MM, min(AXIS_MAX_MM, v))

    def _guard(self, action_name):
        """运动命令门控: 急停态/运动中拒绝。返回True允许。"""
        with self._lock:
            if self._state == STATE_ESTOP:
                self._log(f"⛔ 急停状态, 拒绝[{action_name}], 请先复位")
                return False
            if self._busy:
                self._log(f"⏳ 运动未结束, 拒绝[{action_name}]")
                return False
        return True

    # ---------- 高层接口 ----------
    def move_to(self, x_mm, y_mm):
        """移动到绝对坐标(mm)。运动中拒绝。"""
        if not self._guard("move_to"):
            return False
        x, y = self._clamp(x_mm), self._clamp(y_mm)
        payload = struct.pack('>hh', int(x * MM_TO_UNIT), int(y * MM_TO_UNIT))
        if self._send(CMD_MOVE_XY, payload):
            self._x, self._y = x, y
            self._set_busy()
            return True
        return False

    def move_xy_rel(self, dx_mm, dy_mm):
        """相对当前坐标移动(mm)。"""
        return self.move_to(self._x + dx_mm, self._y + dy_mm)

    def move_z(self, steps):
        """Z轴步进(步数)。运动中拒绝。"""
        if not self._guard("move_z"):
            return False
        if self._send(CMD_MOVE_Z, struct.pack('>h', int(steps))):
            self._set_busy()
            return True
        return False

    def move_xyz(self, x_mm, y_mm, z_steps):
        """XYZ联动移动: 一帧同时下发三轴, STM32端三轴同时启动。运动中拒绝。"""
        if not self._guard("move_xyz"):
            return False
        x, y = self._clamp(x_mm), self._clamp(y_mm)
        payload = struct.pack('>hhh', int(x * MM_TO_UNIT), int(y * MM_TO_UNIT), int(z_steps))
        if self._send(CMD_MOVE_XYZ, payload):
            self._x, self._y = x, y
            self._set_busy()
            return True
        return False

    def home(self):
        """回零。运动中拒绝(急停态允许, 兼做复位)。"""
        with self._lock:
            estop = (self._state == STATE_ESTOP)
            busy = self._busy
        if not estop and busy:
            self._log("⏳ 运动未结束, 拒绝[home]")
            return False
        if self._send(CMD_HOME):
            self._x, self._y = 0.0, 0.0
            with self._lock:
                self._state = STATE_MOVING
                self._busy = True
                self._busy_since = time.time()
            return True
        return False

    def estop(self):
        """急停。任何时候都可发送(安全优先), 立即置急停态。"""
        ok = self._send(CMD_ESTOP)
        with self._lock:
            self._state = STATE_ESTOP
            self._busy = False
        return ok

    def reset(self):
        """复位急停态(通过回零命令0x03, STM32端急停态收到0x03会转IDLE)。"""
        if self._send(CMD_HOME):
            self._x, self._y = 0.0, 0.0
            with self._lock:
                self._state = STATE_MOVING
                self._busy = True
                self._busy_since = time.time()
            return True
        return False

    def squeeze(self, count=1):
        """挤锡count次。运动中拒绝。"""
        if not self._guard("squeeze"):
            return False
        if self._send(CMD_SQUEEZE, struct.pack('>B', int(count) & 0xFF)):
            self._set_busy()
            return True
        return False

    def calibrate_zero(self):
        """绝对零点校准: 将STM32当前物理位置标定为工作坐标(-360,-360)。
        非运动命令(不移动电机,不置busy)。需在回零后调用以保证状态干净。"""
        return self._send(CMD_CALIB_ZERO)

    def get_position(self):
        """返回逻辑坐标(x_mm, y_mm)。"""
        return (self._x, self._y)

    def set_position(self, x_mm, y_mm):
        """手动同步逻辑坐标(不发指令)。"""
        self._x, self._y = x_mm, y_mm


# ===== 自测 =====
if __name__ == '__main__':
    print("帧编码自测:")
    print("  move_to(100,50):", _build_frame(CMD_MOVE_XY, struct.pack('>hh', 1000, 500)).hex())
    print("  home:", _build_frame(CMD_HOME).hex())
    print("  estop:", _build_frame(CMD_ESTOP).hex())
    print("  move_z(-50):", _build_frame(CMD_MOVE_Z, struct.pack('>h', -50)).hex())
    print("  squeeze(1):", _build_frame(CMD_SQUEEZE, struct.pack('>B', 1)).hex())
    # 上行解析自测
    mc = MotorController()
    st = _build_frame(REP_STATUS, struct.pack('>hhh', 1234, 567, 100) + bytes([1]))
    print("  上行状态帧示例:", st.hex())
    mc._handle_report(REP_STATUS, st[3:-1])
    print("  解析后 pos/state:", mc.get_position(), mc.get_state(), "busy:", mc.is_busy())

# ============================================================
# STM32端解析参考(USB CDC回调里调用, 复用TJC_RxCallback状态机):
#   下行已有: 0x01/0x02/0x03 见 TJC_RxCallback
#   下行待加: 0x06 int16 steps→stepper3步进; 0x07 payload[0]→squeeze_paste
#             0x08 int16 x,y,z→三轴同时启动(x,y为0.1mm绝对, z为步数, 同home的并发驱动方式)
#             0x09 无payload→标定当前位置为(-360,-360): off1=m1_phys, off2=m2_phys+509.117
#   上行需加: 周期发 0x10 状态帧; 运动到位发 0x11 完成帧
#     uint8 b[]; b[0]=0xAA;b[1]=0x55;b[2]=0x10;
#     int16 x=realx*10; ... b[3..]=x_hi,x_lo,y_hi,y_lo,z_hi,z_lo,state;
#     b[10]=sum(b[2..9])&0xFF; CDC_Transmit_FS(b,11);
# ============================================================
