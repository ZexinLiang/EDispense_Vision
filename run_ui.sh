#!/bin/bash
export DISPLAY=:0
export QT_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms

# 触摸屏映射到HDMI-1 + 禁用GNOME手势
xinput set-prop "wch.cn USB2IIC_CTP_CONTROL" "Coordinate Transformation Matrix" 0.652174 0 0 0 1 0 0 0 1
xinput set-prop "wch.cn USB2IIC_CTP_CONTROL" "libinput Calibration Matrix" 1 0 0 0 1 0 0 0 1

cd /home/elf/solder_system
python3 ui/solder_ui.py
