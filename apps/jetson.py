"""
VisionLink Jetson - Jetson Orin Nano 边缘部署版
纯自动扫描 + 无头兼容 + 量化模型 + 低分辨率

运行方式: python -m apps.jetson
或从项目根目录: python apps/jetson.py
"""
import os
import sys
import time
import logging
import threading
import select
import termios
import tty

import cv2

# 屏蔽 Qt 字体警告（Jetson 环境常见问题）
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu/")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.platform import IS_JETSON, HAS_DISPLAY
from src.config import (
    CAMERA_CONFIG, STATE_IDLE, SNAPSHOT_DIR, INFER_LOG_DIR, AUDIO_CACHE_DIR,
    MODEL_NAME,
)
from src.prompts import PROMPT_LIB
from src.camera import CameraManager
from src.inference import InferenceEngine
from src.tts import TTSEngine
from src.ui import UIManager

# ==================== 日志配置 ====================
sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="{asctime} | {levelname:8} | {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("jetson")

# ==================== 目录初始化 ====================
for d in [SNAPSHOT_DIR, INFER_LOG_DIR, AUDIO_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 模式与 Prompt ====================
MODE_NAMES = ["障碍物检测", "文字识别", "人脸检测", "场景描述", "图文问答"]

# ==================== 引擎实例 ====================
camera = CameraManager()
infer = InferenceEngine()
tts = TTSEngine()
ui = UIManager()
ui.enable_gui = False  # 强制无头模式，纯终端交互

current_mode = 1  # 默认障碍物检测


def save_snapshot(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SNAPSHOT_DIR, f"snap_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    return path


def save_infer_log(snapshot_path, mode_idx, result):
    timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(INFER_LOG_DIR, f"infer_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模式：{MODE_NAMES[mode_idx - 1]}\n")
        f.write(f"抓拍：{snapshot_path}\n")
        f.write("=" * 40 + "\n")
        f.write(result)
    return path


def run_inference(frame, mode_idx):
    """单次推理任务"""
    snapshot_path = save_snapshot(frame)
    logger.info(f"推理模式 {mode_idx}，抓拍：{snapshot_path}")

    img_b64 = infer.image_to_base64(frame)
    if not img_b64:
        return

    prompt = PROMPT_LIB["zh"][mode_idx - 1]
    result = infer.infer(prompt, img_b64)

    if result:
        log_path = save_infer_log(snapshot_path, mode_idx, result)
        logger.info(f"结果: {result[:80]}...")
        logger.info(f"日志: {log_path}")
        tts.speak(result)


def get_terminal_key():
    """从终端非阻塞读取按键（不依赖 OpenCV 窗口焦点）"""
    if select.select([sys.stdin], [], [], 0.02)[0]:
        return sys.stdin.read(1)
    return ""


def main():
    global current_mode

    logger.info("=" * 50)
    logger.info(f"VisionLink Jetson 启动 | 无头模式: {not HAS_DISPLAY}")
    logger.info(f"模型: {MODEL_NAME}")
    for i, name in enumerate(MODE_NAMES, 1):
        logger.info(f"  模式{i}: {name}")
    logger.info("触发方式: 空格键手动触发（终端键盘，无需窗口焦点）")
    logger.info("操作: 空格=触发 | 1~5=切换模式 | S=停止语音 | Q=退出")
    logger.info(f"存储: {SNAPSHOT_DIR} | {INFER_LOG_DIR} | {AUDIO_CACHE_DIR}")
    logger.info("=" * 50)

    if not camera.open():
        logger.error("摄像头初始化失败")
        return

    tts.speak("项目启动")

    # 设置终端为非规范模式（无需回车即可读取按键）
    use_terminal = sys.stdin.isatty()
    if use_terminal:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        logger.info("终端键盘监听已启用")
    else:
        old_settings = None
        logger.warning("stdin 不是终端，键盘输入不可用，仅支持 Ctrl+C 退出")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.05)
                continue

            # 从终端读取按键（不依赖窗口焦点）
            key = get_terminal_key() if use_terminal else ""

            if not key:
                continue

            # 空格键：触发推理（播放拍照音效）
            if key == " " and not infer.is_busy:
                logger.info("空格触发推理")
                tts.play_effect("assets/audio/pic.mp3")
                threading.Thread(
                    target=run_inference,
                    args=(frame.copy(), current_mode),
                    daemon=True,
                ).start()
                continue

            # 1~5 切换模式
            elif key in "12345":
                tts.play_beep()
                current_mode = int(key)
                tts.speak(f"已切换至{MODE_NAMES[current_mode - 1]}")
                continue

            # S/s 停止语音
            elif key.lower() == "s":
                tts.play_beep()
                tts.stop()
                continue

            # Q/q 退出
            elif key.lower() == "q":
                tts.play_beep()
                break

    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        camera.release()
        ui.destroy()
        tts.stop()
        logger.info("程序结束")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"全局异常：{e}")
        raise
