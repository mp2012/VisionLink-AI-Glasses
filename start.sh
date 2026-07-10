#!/bin/bash
# VisionLink 快速启动脚本
# 用法: ./start.sh [模式]
#   默认:      ./start.sh            → 单摄像头 POV
#   双摄:      ./start.sh dual       → POV + FOV 双摄像头
#   全功能:    ./start.sh full       → 双摄 + YOLO 避障
#   桌面:      ./start.sh desktop    → GUI 桌面模式

set -e
cd "$(dirname "$0")"

# 激活虚拟环境（如果存在）
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    echo "[OK] 虚拟环境已激活"
fi

# 设置默认音频输出为 AB13X USB 耳机（如果 PulseAudio 正在运行）
if command -v pactl &>/dev/null && pactl info &>/dev/null 2>&1; then
    AB13X_SINK="alsa_output.usb-Generic_AB13X_USB_Audio_20210726905926-00.iec958-stereo"
    CURRENT_SINK=$(pactl get-default-sink 2>/dev/null)
    if [ "$CURRENT_SINK" != "$AB13X_SINK" ]; then
        pactl set-default-sink "$AB13X_SINK" 2>/dev/null && \
            echo "[OK] 音频输出 → AB13X USB 耳机"
    else
        echo "[OK] 音频输出已配置 → AB13X USB 耳机"
    fi
fi

MODE="${1:-default}"

case "$MODE" in
    dual)
        echo "启动 VisionLink - 双摄像头模式..."
        python apps/headless.py --dual
        ;;
    full)
        echo "启动 VisionLink - 全功能模式（双摄 + YOLO 避障）..."
        python apps/headless.py --dual --yolo
        ;;
    desktop)
        echo "启动 VisionLink - 桌面 GUI 模式..."
        python apps/desktop.py
        ;;
    gui)
        echo "启动 VisionLink - 无头模式 + GUI 调试窗口..."
        python apps/headless.py --gui
        ;;
    *)
        echo "启动 VisionLink - 默认模式（单摄像头 POV）..."
        python apps/headless.py
        ;;
esac
