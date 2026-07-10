"""
VisionLink Windows 桌面版入口
全功能 GUI 模式：5 种功能模式 + 中英双语 + 自动模式 + UI 面板

启动方式:
    python apps/desktop.py
"""
import os
import sys
import time
import logging
import threading

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.platform import IS_JETSON, IS_WINDOWS, HAS_DISPLAY
from src.config import MODE_NAMES, SNAPSHOT_DIR, INFER_LOG_DIR, AUDIO_CACHE_DIR
from src.camera import CameraManager
from src.inference import InferenceEngine
from src.tts import TTSEngine
from src.agent import Agent
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
logger = logging.getLogger("desktop")

# ==================== 目录初始化 ====================
for d in [SNAPSHOT_DIR, INFER_LOG_DIR, AUDIO_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 引擎实例 ====================
camera = CameraManager(name="POV-Camera")
infer = InferenceEngine()
tts = TTSEngine()
ui = UIManager(enable_gui=True)
agent = Agent(infer, tts, camera)

current_mode = 1
voice_lang = "zh"


def main():
    global current_mode, voice_lang

    logger.info("=" * 55)
    logger.info("VisionLink Desktop 启动")
    for i, name in enumerate(MODE_NAMES, 1):
        logger.info(f"  模式{i}: {name}")
    logger.info("操作: 空格=触发 | 1~5=切换模式 | L=切换语种 | M=自动 | S=停止 | ESC=退出")
    logger.info("=" * 55)

    if not camera.open():
        logger.error("摄像头初始化失败")
        return

    tts.speak("项目启动")

    # GUI 窗口
    cv2.namedWindow("VisionLink Desktop", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()

            # 自动模式
            if agent.should_scan(now):
                threading.Thread(
                    target=agent.auto_scan,
                    args=(frame.copy(),),
                    daemon=True
                ).start()

            # 渲染 UI
            display = ui.render(
                frame, agent.state, agent.current_mode,
                auto_enabled=agent.auto_enabled,
                lang=voice_lang
            )
            cv2.imshow("VisionLink Desktop", display)

            # 键盘处理
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            # 空格
            if key == 32:
                agent.handle_trigger(frame)
                continue

            # 1~5
            elif ord("1") <= key <= ord("5"):
                agent.set_mode(key - ord("0"))
                continue

            # L: 切换语种
            elif key == ord("l") or key == ord("L"):
                voice_lang = "en" if voice_lang == "zh" else "zh"
                agent.set_lang(voice_lang)
                continue

            # M: 自动模式
            elif key == ord("m") or key == ord("M"):
                agent.toggle_auto()
                continue

            # S: 停止语音
            elif key == ord("s") or key == ord("S"):
                tts.stop()
                continue

            # ESC
            elif key == 27:
                break

    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        camera.release()
        ui.destroy()
        tts.stop()
        agent.shutdown()
        logger.info("程序结束")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"全局异常: {e}")
        raise
