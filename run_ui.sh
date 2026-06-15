#!/bin/bash
export DISPLAY=:0
export QT_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms
cd /home/elf/solder_system/ui
exec python3 solder_ui.py
