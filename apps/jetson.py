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

# 屏蔽 Qt 字体警告（Jetson 环境常见问题）
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu/")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.platform import IS_JETSON, HAS_DISPLAY
from src.config import (
    CAMERA_CONFIG, STATE_IDLE, SNAPSHOT_DIR, INFER_LOG_DIR, AUDIO_CACHE_DIR,
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

current_mode = 1  # 默认障碍物检测
scan_interval = 4  # 秒


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


def main():
    global current_mode

    logger.info("=" * 50)
    logger.info(f"VisionLink Jetson 启动 | 无头模式: {not HAS_DISPLAY}")
    logger.info(f"模型: {InferenceEngine.__module__}")
    logger.info(f"扫描间隔: {scan_interval}s")
    logger.info(f"存储: {SNAPSHOT_DIR} | {INFER_LOG_DIR} | {AUDIO_CACHE_DIR}")
    logger.info("=" * 50)

    tts.speak("系统启动完成，自动扫描模式")

    if not camera.open():
        logger.error("摄像头初始化失败")
        return

    if HAS_DISPLAY:
        win_name = "VisionLink Jetson"
        ui.create_window(win_name, CAMERA_CONFIG["width"], CAMERA_CONFIG["height"])

    last_scan = 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.05)
                continue

            now = time.time()

            # 自动扫描
            if now - last_scan >= scan_interval and not infer.is_busy:
                last_scan = now
                logger.info("自动扫描触发")
                threading.Thread(
                    target=run_inference,
                    args=(frame.copy(), current_mode),
                    daemon=True,
                ).start()

            # UI 显示
            if ui.enable_gui:
                display = ui.draw_panel(frame)
                ui.show(win_name, display)
                key = cv2.waitKey(20) & 0xFF

                # 1~5 切换模式
                if 49 <= key <= 53:
                    tts.play_beep()
                    current_mode = key - 48
                    tts.speak(f"已切换至{MODE_NAMES[current_mode - 1]}")
                    continue

                # S 停止语音
                if key == ord('s') or key == ord('S'):
                    tts.play_beep()
                    tts.stop()
                    continue

                # ESC 退出
                if key == 27:
                    break
            else:
                # 无头模式纯休眠
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
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
