"""
VisionLink Jetson - 边缘部署版（兼容入口）
直接复用 Agent 架构，保持向后兼容

运行方式: python apps/jetson.py
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

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu/")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.platform import IS_JETSON, HAS_DISPLAY
from src.config import MODE_NAMES, SNAPSHOT_DIR, INFER_LOG_DIR, AUDIO_CACHE_DIR, AUDIO_DEVICE
from src.camera import CameraManager
from src.inference import InferenceEngine
from src.tts import TTSEngine
from src.agent import Agent
from src.prompts import TIP_VOICE
from src.ui import UIManager
from src.volume_control import VolumeController

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

for d in [SNAPSHOT_DIR, INFER_LOG_DIR, AUDIO_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 引擎实例 ====================
camera = CameraManager(name="POV-镜腿单目")
infer = InferenceEngine()
tts = TTSEngine()
ui = UIManager()
ui.enable_gui = False
vol_ctrl = VolumeController(audio_device=AUDIO_DEVICE)
agent = Agent(infer, tts, camera)


def get_terminal_key():
    """非阻塞读取终端按键"""
    if select.select([sys.stdin], [], [], 0.02)[0]:
        return sys.stdin.read(1)
    return ""


def main():
    logger.info("=" * 50)
    logger.info(f"VisionLink Jetson 启动 | 无头模式: {not HAS_DISPLAY}")
    logger.info(f"模型: {infer.model}")
    for i, name in enumerate(MODE_NAMES, 1):
        logger.info(f"  模式{i}: {name}")
    logger.info("触发方式: 空格键手动触发（终端键盘，无需窗口焦点）")
    logger.info("操作: 空格=触发 | 1~5=切换模式 | S=停止语音 | Q=退出")
    logger.info(f"存储: {SNAPSHOT_DIR} | {INFER_LOG_DIR} | {AUDIO_CACHE_DIR}")
    logger.info("=" * 50)

    if not camera.open():
        logger.error("摄像头初始化失败")
        return

    tts.speak(TIP_VOICE["zh"]["start"])
    vol_ctrl.start()

    use_terminal = sys.stdin.isatty()
    if use_terminal:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        logger.info("终端键盘监听已启用")
    else:
        old_settings = None
        logger.warning("stdin 不是终端，键盘输入不可用")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.05)
                continue

            key = get_terminal_key() if use_terminal else ""

            if not key:
                continue

            # 空格键：拍照推理
            if key == " ":
                agent.handle_trigger(frame)
                continue

            # 1~5 切换模式
            elif key in "12345":
                tts.play_beep()
                agent.set_mode(int(key))
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
        vol_ctrl.stop()
        agent.shutdown()
        logger.info("程序结束")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"全局异常: {e}")
        raise
