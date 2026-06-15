# EDispense_Vision

基于AI视觉与热工艺协同的高一致性点锡与焊接检查辅助系统

## 平台
- 硬件：飞凌ELF2 (RK3588) 开发板
- 系统：Ubuntu 22.04 aarch64
- 推理：RKNNLite (YOLOv5n)

## 架构
```
solder_system/
├── ui/solder_ui.py          # PyQt5 触摸UI (1024x600自适应)
├── vision/
│   ├── infer.py             # RKNN YOLOv5n 焊盘检测
│   └── path_generator.py    # 点锡路径规划 + G-code生成
├── models/
│   └── material-640-640-v5n.rknn  # 焊盘检测模型
├── collect.py               # 数据集采集工具
├── run_ui.sh                # 启动脚本
├── data/                    # 测试图片
├── datasheet/               # 采集数据
└── output/                  # 推理输出(路径JSON/G-code/可视化)
```

## 功能
1. **视觉检测**：摄像头采集PCB图像，YOLOv5n实时检测焊盘位置
2. **路径规划**：小焊盘单点点锡，大焊盘网格填充，贪心TSP路径优化
3. **G-code输出**：生成点锡机运动控制指令
4. **触摸UI**：PyQt5全屏界面，支持参数调节和实时预览

## 运行
```bash
./run_ui.sh
```
