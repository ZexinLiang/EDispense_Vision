# EDispense_Vision

**English** | [中文](#edispense_vision-中文)

An AI-vision and thermal-process co-driven, high-consistency solder-paste dispensing and AOI (Automated Optical Inspection) assistant system.

It runs on a Forlinx ELF2 (RK3588) board and works together with a self-developed XYZ 3-axis dispensing machine (STM32 lower-computer) to automatically locate PCB pads and dispense solder paste.

## Table of Contents

- [Platform and Hardware](#platform-and-hardware)
- [Directory Structure](#directory-structure)
- [Working Modes](#working-modes)
- [Features](#features)
- [Lower-Computer Protocol (USB CDC)](#lower-computer-protocol-usb-cdc)
- [Running](#running)
- [Key Parameters](#key-parameters)
- [Notes and Pitfalls](#notes-and-pitfalls)
- [Development Workflow](#development-workflow)
- [License](#license)

## Platform and Hardware

| Item | Description |
|------|-------------|
| SoC | Forlinx ELF2 (RK3588), Ubuntu 22.04 aarch64 |
| Inference | RKNNLite (YOLOv5s), NPU-accelerated |
| Display | 1024x600 HDMI touch screen (wch.cn USB2IIC capacitive touch) |
| Lower computer | STM32, **USB CDC virtual serial port** (not a physical UART), drives XYZ 3-axis + solder extrusion |
| Cameras | Top camera (video21, XY positioning) + side camera (video23, Z-height calibration) |

The detection model is a YOLOv5s network converted to RKNN and running on the RK3588 NPU. The inference input is **1088x1088**: a 1920x1080 MJPG frame is center-cropped to a 1080x1080 square ROI, then resized directly to 1088x1088 (BGR to RGB) before being fed to the NPU. Detection coordinates are mapped back to the original image space.

## Directory Structure

```
solder_system/
├── ui/solder_ui.py          # PyQt5 touch main program (all interaction logic / execution state machine)
├── vision/
│   ├── infer.py             # RKNN local inference (INPUT_SIZE = 1088)
│   ├── remote_infer.py      # External inference client (connects to the Win11 inference server)
│   └── path_generator.py    # Dispensing path planning + G-code generation
├── models/                  # RKNN models (pad.rknn / qs.rknn)
├── config/
│   ├── xy_calib.json        # XY three-point affine calibration matrix
│   ├── z_plane.json         # Z-plane four-point least-squares calibration
│   └── calibration_result_4.npz
├── gerber_paste_parser.py   # Parses Gerber solder-paste layer (.GTP/.GBP) into mm coordinates
├── gerber_upload_server.py  # HTTP service for wireless Gerber upload (auto-start on boot)
├── motor_control.py         # Lower-computer command wrapper
├── collect.py / collect_dataset.py  # Dataset collection
├── run_ui.sh                # Launch script (sets DISPLAY / touch mapping / starts UI)
└── output/                  # Path JSON / G-code / visualization images
```

## Working Modes

The program has two top-level modes (switchable from the top bar):

- **Solder mode**: detect pads to generate a dispensing path to execute dispensing with the 3 axes.
- **AOI mode**: load or capture an image for solder-quality inspection.

The dispensing path has two mutually exclusive sources:

1. **Vision path**: the camera runs real-time inference to detect pads and generates the path directly.
2. **Gerber path**: import the PCB Gerber solder-paste layer, overlay the mask on the live view, align it, then generate the path from Gerber pad coordinates (more precise, independent of detection recall).

## Features

### Vision detection and path generation
- Real-time camera inference detects pads in the frame.
- Clicking **Generate Path** performs a **frame-lock**: the view freezes on the current frame's path result while the camera and inference threads **keep running** (not stopped); the inference toggle switches to "Resume Inference". Clicking "Resume Inference" unfreezes back to live.
- After path generation the UI enters an **edit state**: clicking any pad box **toggles select/deselect** (deselected pads are not dispensed). This works whether you "lock inference first then generate" or "generate directly during live inference".
- After changing the selection, click **Generate Path** again to recompute the path for the new selection.

### Path planning algorithm (`vision/path_generator.py`)
- **Large/small pad classification is by width and height** (not area), threshold `SINGLE_PAD_MAX_DIM = 45` pixels:
  - Both width and height < threshold: small pad, a **single center point**.
  - Either side >= threshold: large pad, a **centered symmetric grid** of multiple fill points.
  - Long strips (e.g. 100x18): points are laid along the long edge, not just the center.
- **Adjustable fill spacing**: the "Fill Spacing" stepper on the UI (5-50 px, default 15) takes effect in real time; smaller means denser, larger means sparser.
- The grid is **symmetrically centered** on the pad's geometric center (equal margins on both sides).
- Greedy nearest-neighbor TSP path optimization with a serpentine route to reduce idle travel; points of the same pad stay grouped.

### Gerber alignment (two methods)
After importing the Gerber solder-paste layer, the pad mask overlays the live view. Two alignment methods are provided:

**A. Point-pair alignment (recommended, default)**
1. Click a Gerber pad on the mask to select it as the source point (yellow circle).
2. Click the corresponding **real pad center** in the view to drop the target point (magenta cross), which **can be re-clicked to override and correct**.
3. When satisfied, click **Lock This Pair** to store one pair (green line + index).
4. Collect **2-3 pairs**, then click **Compute Alignment** to auto-fit a similarity transform (translation + rotation + uniform scale), snapping the whole mask into place.
5. The undo arrow removes the last pair. A residual > 15 px warns of a possible misclick.

**B. Drag fine-tuning (auxiliary)**
- After enabling the "Template Ops" toggle, use two fingers on the touch screen to rotate/scale/translate the mask for fine adjustment.

After alignment, click **Confirm Alignment** to lock the pose, then click Generate Path. You can click individual pads to **exclude** them (not dispensed). **Cancel Alignment** exits and clears the mask.

### 3-axis dispensing execution
- Queue-driven state machine (50ms timer), flow:
  **Home to zero → [prime x3] → [XY to point → Z down to contact → extrude → Z lift to safe] x N → home to zero**.
- **Post-reset priming**: after homing, extrude solder paste 3 times in place (`SOLDER_PRIME_COUNT=3`) so the first real pad dispenses correctly. Each extrusion waits `SOLDER_PRIME_GAP_MS=500ms` **after completion** (confirmed by the lower-computer busy flag clearing) before the next.
- Z uses **absolute position tracking**, so resuming after a pause-and-home does not desync.
- Three-state control: pause / resume / abort. `is_busy()` gating prevents needle crashes.

### Calibration
- **XY calibration**: top-camera three-point affine (`cv2.getAffineTransform`), stored in `config/xy_calib.json`.
- **Z calibration**: side camera, four-point least-squares fit of the dispensing plane, stored in `config/z_plane.json`; the target Z at each (x,y) is interpolated at execution time.

### Others
- **Wireless Gerber upload**: `gerber_upload_server.py` auto-starts on boot; you can upload a Gerber archive from another device's web page to the SD card at `/media/elf/OPI_BOOT/Gerber`.
- **External inference**: `vision/remote_infer.py` can switch to a Win11 inference server when the board NPU is insufficient.

## Lower-Computer Protocol (USB CDC)

Downlink frame format: `AA 55 | CMD | payload | SUM`

| CMD | Function |
|-----|----------|
| 0x01 | XY absolute move |
| 0x02 | Emergency stop |
| 0x03 | Home / reset |
| 0x06 | Z step (+down / -up) |
| 0x07 | Extrude (payload = count) |
| 0x08 | XYZ combined move |
| 0x09 | Zero-point calibration |

> `StpDistanceSetBlocking` is actually non-blocking, so the three axes can run concurrently. Command completion is judged by reading the **busy status flag** plus a minimum wait time.

## Running

```bash
./run_ui.sh
```

The script sets `DISPLAY=:0`, the Qt plugin path, and the touch-screen coordinate mapping, then launches `ui/solder_ui.py`.

## Key Parameters

`ui/solder_ui.py` top-level constants:

| Constant | Default | Meaning |
|----------|---------|---------|
| `SOLDER_Z_DOWN_STEPS` | 890 | Z down steps when dispensing (contact the pad) |
| `SOLDER_Z_LIFT_POS` | 600 | Z lift safe position between points |
| `SOLDER_DISP_CROP_W` | 1080 | Display crop width in solder mode (square, centered); offset auto-follows |
| `SOLDER_SQUEEZE_COUNT` | 1 | Number of squeezes per single extrusion |
| `SOLDER_PRIME_COUNT` | 3 | In-place priming extrusions after reset |
| `SOLDER_PRIME_GAP_MS` | 500 | Gap between priming extrusions (ms) |
| `SOLDER_XY_MIN_WAIT_MS` | 1200 | Minimum wait for XY move |
| `SOLDER_HOME_TIMEOUT_MS` | 20000 | Homing timeout |

`vision/path_generator.py` top-level:

| Constant | Default | Meaning |
|----------|---------|---------|
| `SINGLE_PAD_MAX_DIM` | 45 | Large/small pad threshold (px, by width/height): either side >= this means multi-point fill |
| `FILL_SPACING` | 15 | Default multi-point fill spacing (px, overridable by UI) |

`vision/infer.py` top-level:

| Constant | Default | Meaning |
|----------|---------|---------|
| `INPUT_SIZE` | 1088 | RKNN model input size (1088x1088) |
| `CONF_THRESH` | 0.25 | Object/class confidence threshold |

## Notes and Pitfalls

- **Coordinate-system pitfall**: the solder-mode view is a horizontally centered crop of the full image (cropping off `disp_offset_x`), while XY calibration is based on the **full image**. At execution, the path point's x must add back `disp_offset_x` to align with calibration. Both the vision path and Gerber path already handle this; when adding any new "display coordinate to machine coordinate" path source, remember to add this offset, otherwise dispensing positions shift by roughly one crop amount.
- **Priming is a "post-completion" delay**: the 0.5s gap counts from the moment extrusion actually completes (busy cleared), not from when the command is sent, so it is unaffected by extrusion duration.
- **Needle-crash protection**: `is_busy()` gating during execution. Manual moves follow "lift Z above the fixed seat first → XY translate → then lower Z"; low-position diagonal moves are forbidden.
- **Closing the window triggers an emergency-stop lock**: normally closing the UI also sends an emergency stop, so the next boot may require "Home" first to unlock before moving.
- **Gerber alignment confirm vs cancel**: "Confirm Alignment" keeps the pad data and uses the Gerber path; "Cancel Alignment" clears the mask and pad data (`_gerber_pads_mm`), otherwise path generation stays blocked by the Gerber branch.
- **Back up coordinate/model/calibration files before changing them**; once a calibration is overwritten it cannot be recovered.
- **WiFi (AX210)**: if the driver hangs on a PNVM timeout, moving the pnvm file aside works around it.
- **STM32 firmware**: built and flashed with Keil on the Windows dev machine; this repository does not include the firmware project.

## Development Workflow

Board-side code lives in `/home/elf/solder_system`. Typical change flow:

1. Edit locally, then `python3 -m py_compile` to check syntax.
2. `scp` / `sftp` to the corresponding directory on the board.
3. Restart the UI: `pkill -f 'ui/solder_ui.py'`, then `setsid bash run_ui.sh >/tmp/solder_ui.log 2>&1 </dev/null &`.
4. Verify on the machine, then `git commit` / `push`.

## License

Released under the [MIT License](LICENSE). Third-party components (RKNN Toolkit / RKNNLite, YOLOv5, OpenCV, PyQt5, etc.) remain under their respective licenses.

---

# EDispense_Vision (中文)

[English](#edispense_vision) | **中文**

基于 AI 视觉与热工艺协同的高一致性点锡 / 焊接检查（AOI）辅助系统。

运行于飞凌 ELF2（RK3588）开发板，配合自研 XYZ 三轴点锡机（STM32 下位机）完成 PCB 焊盘的自动定位与点锡。

## 目录

- [平台与硬件](#平台与硬件)
- [目录结构](#目录结构)
- [工作模式](#工作模式)
- [功能详解](#功能详解)
- [下位机通信协议-usb-cdc](#下位机通信协议-usb-cdc)
- [运行](#运行)
- [关键参数](#关键参数)
- [注意事项](#注意事项)
- [部署--开发流程](#部署--开发流程)
- [许可协议](#许可协议)

## 平台与硬件

| 项目 | 说明 |
|------|------|
| 主控 | 飞凌 ELF2 (RK3588)，Ubuntu 22.04 aarch64 |
| 推理 | RKNNLite (YOLOv5s)，NPU 加速 |
| 显示 | 1024x600 HDMI 触摸屏（wch.cn USB2IIC 电容触控） |
| 下位机 | STM32，**USB CDC 虚拟串口**（非物理串口）通信，驱动 XYZ 三轴 + 挤锡 |
| 相机 | 顶视相机（video21，管 XY 定位）+ 侧视相机（video23，管 Z 高度标定） |

检测模型为 YOLOv5s 网络转成 RKNN 后运行在 RK3588 NPU 上。推理输入为 **1088x1088**：1920x1080 MJPG 采集帧中心裁剪为 1080x1080 正方形 ROI，再直接 resize 到 1088x1088（BGR 转 RGB）送入 NPU，检测坐标再映射回原图空间。

## 目录结构

```
solder_system/
├── ui/solder_ui.py          # PyQt5 触摸主程序（全部交互逻辑 / 执行状态机）
├── vision/
│   ├── infer.py             # RKNN 本地推理（INPUT_SIZE = 1088）
│   ├── remote_infer.py      # 外部推理客户端（连 Win11 推理服务端）
│   └── path_generator.py    # 点锡路径规划 + G-code 生成
├── models/                  # RKNN 模型（pad.rknn / qs.rknn）
├── config/
│   ├── xy_calib.json        # XY 三点仿射标定矩阵
│   ├── z_plane.json         # Z 平面四点最小二乘标定
│   └── calibration_result_4.npz
├── gerber_paste_parser.py   # Gerber 锡膏层(.GTP/.GBP)解析为 mm 坐标
├── gerber_upload_server.py  # 无线上传 Gerber 的 HTTP 服务（开机自启）
├── motor_control.py         # 下位机命令封装
├── collect.py / collect_dataset.py  # 数据采集
├── run_ui.sh                # 启动脚本（设 DISPLAY / 触摸映射 / 启 UI）
└── output/                  # 路径 JSON / G-code / 可视化图
```

## 工作模式

程序分两大模式（顶部切换）：

- **点锡模式 (solder)**：检测焊盘 → 生成点锡路径 → 三轴执行点锡。
- **AOI 模式**：加载 / 抓取图像做焊接质量检查。

点锡路径有两条来源，互斥：

1. **视觉路径**：摄像头实时推理检测焊盘，直接生成路径。
2. **Gerber 路径**：导入 PCB 的 Gerber 锡膏层，蒙版叠加到实时画面，对位后按 Gerber 焊盘坐标生成路径（坐标更精确、不依赖检测召回）。

## 功能详解

### 视觉检测与路径生成
- 摄像头实时推理，检测画面中的焊盘。
- 点「**路径生成**」= 一次**锁帧**操作：画面冻结在当前帧的路径结果上，摄像头与推理线程**继续运行**（不关闭），推理开关按钮切到「继续推理」。点「继续推理」解冻回实时。
- 路径生成后进入**编辑态**：点击任意焊盘框可**反选 / 恢复**（被反选的焊盘不点锡）。无论是「先锁定推理→再生成」还是「动态推理中直接生成」，都能反选。
- 反选改变后需**再点一次「路径生成」**按新选择重算路径。

### 路径规划算法 (`vision/path_generator.py`)
- **大 / 小焊盘判据按长宽**（非面积），阈值 `SINGLE_PAD_MAX_DIM = 45` 像素：
  - 长宽**都** < 阈值 → 小焊盘，**中心单点**。
  - **任一边** >= 阈值 → 大焊盘，**居中对称网格**多点填充。
  - 细长条（如 100x18）→ 沿长边铺一排点，不会只点中心。
- **填充间距可调**：UI 上「填充间距」步进器（5~50 像素，默认 15），实时生效；调小点更密、调大点更稀。
- 网格点阵关于焊盘几何中心**对称居中**（两侧留白相等）。
- 贪心最近邻 TSP 优化路径，蛇形走位减少空行程，同焊盘的点分组不打散。

### Gerber 对位（两种方式）
导入 Gerber 锡膏层后，焊盘蒙版叠加到实时画面，提供两种对位手段：

**A. 点对对位（推荐，默认进入）**
1. 点蒙版上的某个 Gerber 焊盘 → 选为源点（黄圈）。
2. 点画面里它对应的**实物焊盘中心** → 落目标点（品红十字），**可反复点击覆盖修正**（一次点不准没关系）。
3. 满意后点「**锁定本对**」→ 存为一对（绿连线 + 序号）。
4. 攒 **2~3 对** → 点「**计算对位**」→ 自动用相似变换（平移+旋转+等比缩放）拟合，蒙版整体吸附到位。
5. 「↩」可撤销最后一对。残差 > 15px 会提示可能点错。

**B. 拖拽微调（辅助）**
- 「模板操作」开关打开后，双指在触摸屏上旋转 / 缩放 / 平移蒙版做精细微调。

对位满意后点「**确认对位**」锁定位姿，再点「路径生成」。可点选单个焊盘**排除**（不点锡）。「**取消对位**」退出并清除蒙版。

### 三轴执行点锡
- 队列驱动状态机（50ms 定时器），流程：
  **开始回零 → [预热挤锡 x3] → [XY 到点 → Z 下降到接触 → 挤锡 → Z 抬到安全位] x N → 结束回零**。
- **复位后预热挤锡**：回零后在原地连挤 3 次锡膏（`SOLDER_PRIME_COUNT=3`），保证第一个真实焊盘能正常出锡。每次挤锡**完成后**（读下位机 busy 标志位清除确认）再间隔 `SOLDER_PRIME_GAP_MS=500ms` 才发下一次。
- Z 用**绝对位追踪**，暂停回零后继续也不会错乱。
- 三态控制：暂停 / 继续 / 终止。`is_busy()` 门控防止撞针。

### 标定
- **XY 标定**：顶视相机三点仿射（`cv2.getAffineTransform`），结果存 `config/xy_calib.json`。
- **Z 标定**：侧视相机，四点最小二乘拟合点锡平面，结果存 `config/z_plane.json`，执行时按 (x,y) 插值出下降目标 Z。

### 其它
- **Gerber 无线上传**：`gerber_upload_server.py` 开机自启，可从其它设备网页上传 Gerber 压缩包到 SD 卡 `/media/elf/OPI_BOOT/Gerber`。
- **外部推理**：`vision/remote_infer.py` 可切换到 Win11 推理服务端（板端 NPU 不足时）。

## 下位机通信协议 (USB CDC)

下行帧格式：`AA 55 | CMD | payload | SUM`

| CMD | 功能 |
|-----|------|
| 0x01 | XY 绝对移动 |
| 0x02 | 急停 |
| 0x03 | 回零 / 复位 |
| 0x06 | Z 步进（+下降 / -上抬） |
| 0x07 | 挤锡（payload = 次数） |
| 0x08 | XYZ 联动 |
| 0x09 | 零点校准 |

> `StpDistanceSetBlocking` 实际为非阻塞，三轴可并发。命令完成判定靠读 **busy 状态标志位** + 最小等待时间。

## 运行

```bash
./run_ui.sh
```

脚本会设置 `DISPLAY=:0`、Qt 插件路径、触摸屏坐标映射，然后启动 `ui/solder_ui.py`。

## 关键参数

`ui/solder_ui.py` 顶部常量：

| 常量 | 默认 | 含义 |
|------|------|------|
| `SOLDER_Z_DOWN_STEPS` | 890 | 点锡时 Z 下降步数（接触焊盘） |
| `SOLDER_Z_LIFT_POS` | 600 | 点间 Z 抬起安全位 |
| `SOLDER_DISP_CROP_W` | 1080 | 点锡模式显示裁剪宽（正方形，居中），偏移自动跟随 |
| `SOLDER_SQUEEZE_COUNT` | 1 | 单次挤锡的挤压次数 |
| `SOLDER_PRIME_COUNT` | 3 | 复位后原地预热挤锡次数 |
| `SOLDER_PRIME_GAP_MS` | 500 | 预热每次挤完到下次的间隔（ms） |
| `SOLDER_XY_MIN_WAIT_MS` | 1200 | XY 移动最小等待 |
| `SOLDER_HOME_TIMEOUT_MS` | 20000 | 回零超时 |

`vision/path_generator.py` 顶部：

| 常量 | 默认 | 含义 |
|------|------|------|
| `SINGLE_PAD_MAX_DIM` | 45 | 大 / 小焊盘判据（像素，按长宽）：任一边 >= 此值即多点填充 |
| `FILL_SPACING` | 15 | 多点填充间距默认值（像素，可被 UI 覆盖） |

`vision/infer.py` 顶部：

| 常量 | 默认 | 含义 |
|------|------|------|
| `INPUT_SIZE` | 1088 | RKNN 模型输入尺寸（1088x1088） |
| `CONF_THRESH` | 0.25 | 目标 / 类别置信度阈值 |

## 注意事项

- **坐标系坑**：点锡模式画面是从全图横向居中裁剪（裁掉 `disp_offset_x`）的，而 XY 标定基于**全图**坐标。执行时路径点的 x 必须加回 `disp_offset_x` 才能与标定对齐。视觉路径与 Gerber 路径都已处理；新增任何"显示坐标→机器坐标"的路径来源时务必补上这个偏移，否则点锡位置会整体偏移约一个裁剪量。
- **预热挤锡是"完成后"延时**：间隔 0.5s 从挤锡真正完成（busy 清除）那一刻算起，不是从发送算起，所以不受挤锡耗时影响。
- **防撞针**：执行中 `is_busy()` 门控。手动移动遵循"先抬 Z 过固定座 → XY 平移 → 再降 Z"，禁止低位斜向联动。
- **关窗会触发急停锁定**：正常关闭 UI 也会发急停，下次开机可能需要先「回原点」解锁才能动。
- **Gerber 对位 confirm vs cancel**：「确认对位」保留焊盘数据走 Gerber；「取消对位」会清空蒙版与焊盘数据（`_gerber_pads_mm`），否则路径生成会一直被 Gerber 分支拦截。
- **改坐标 / 模型 / 标定文件前先备份**，标定一旦覆盖无法恢复。
- **WiFi (AX210)**：若驱动加载卡 PNVM 超时，移走 pnvm 文件可绕过。
- **STM32 固件**：在开发机 Windows 侧用 Keil 编译烧录，本仓库不含固件工程。

## 部署 / 开发流程

板端代码在 `/home/elf/solder_system`。常规改动流程：

1. 本地改 → `python3 -m py_compile` 校验语法。
2. `scp` / `sftp` 传到板子对应目录。
3. 重启 UI：`pkill -f 'ui/solder_ui.py'`，再 `setsid bash run_ui.sh >/tmp/solder_ui.log 2>&1 </dev/null &`。
4. 联机验证后再 `git commit` / `push`。

## 许可协议

本项目基于 [MIT 许可协议](LICENSE) 发布。第三方组件（RKNN Toolkit / RKNNLite、YOLOv5、OpenCV、PyQt5 等）仍遵循各自的许可协议。
