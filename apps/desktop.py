"""
VisionLink Desktop - Windows 桌面全功能版
5 种模式 + 中英双语 + 语音交互 + UI 面板 + 自动模式

运行方式: python -m apps.desktop
或从项目根目录: python apps/desktop.py
"""
import os
import sys
import time
import logging
import threading

import cv2

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    CAMERA_CONFIG, SPACE_DEBOUNCE, SAVE_DIR,
)
from src.prompts import MODE_NAME_LIST, PROMPT_LIB, TIP_VOICE
from src.camera import CameraManager
from src.inference import InferenceEngine
from src.tts import TTSEngine
from src.ui import UIManager
from src.agent import AutoAgent

# ==================== 日志配置 ====================
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
logging.getLogger("comtypes").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} | {levelname} | {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("desktop")


def zh2gbk(s):
    return s.encode("gbk", errors="replace").decode("gbk")


# ==================== 图片保存 ====================
os.makedirs(SAVE_DIR, exist_ok=True)


def save_clean_frame(frame, mode_num):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"mode_{mode_num}_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)
    logger.info(f"图片已保存: {filepath}")


# ==================== 全局状态 ====================
current_mode = 1
voice_lang = "zh"
auto_enabled = False
last_space_press = 0

# 引擎实例
camera = CameraManager()
infer = InferenceEngine()
tts = TTSEngine()
ui = UIManager()
agent = AutoAgent(infer, tts)

# 功能映射（需要在 agent 初始化后绑定）
def func_obstacle(frame):
    logger.info("执行：障碍物检测")
    save_clean_frame(frame, 1)
    img_b64 = infer.image_to_base64(frame)
    res = infer.infer(PROMPT_LIB[voice_lang][0], img_b64)
    if res:
        tts.speak(res)

def func_ocr(frame):
    logger.info("执行：文字识别")
    save_clean_frame(frame, 2)
    img_b64 = infer.image_to_base64(frame)
    res = infer.infer(PROMPT_LIB[voice_lang][1], img_b64)
    if res:
        tts.speak(res)

def func_face(frame):
    logger.info("执行：人脸检测")
    save_clean_frame(frame, 3)
    img_b64 = infer.image_to_base64(frame)
    res = infer.infer(PROMPT_LIB[voice_lang][2], img_b64)
    if res:
        tts.speak(res)

def func_scene(frame):
    logger.info("执行：场景介绍")
    save_clean_frame(frame, 4)
    img_b64 = infer.image_to_base64(frame)
    res = infer.infer(PROMPT_LIB[voice_lang][3], img_b64)
    if res:
        tts.speak(res)

def func_chat(frame):
    logger.info("执行：语音交互")
    save_clean_frame(frame, 5)
    agent.set_lang(voice_lang)
    agent.handle_voice_chat(frame)


FUNC_MAP = {
    1: func_obstacle,
    2: func_ocr,
    3: func_face,
    4: func_scene,
    5: func_chat,
}


def switch_language():
    global voice_lang
    if voice_lang == "zh":
        voice_lang = "en"
        tts.speak(TIP_VOICE["en"]["lang_switch_en"])
    else:
        voice_lang = "zh"
        tts.speak(TIP_VOICE["zh"]["lang_switch_zh"])
    logger.info(f"语种切换为：{voice_lang}")


def main():
    global current_mode, auto_enabled, last_space_press

    logger.info("=" * 50)
    logger.info("VisionLink Desktop 启动")
    logger.info("=" * 50)
    tts.speak(TIP_VOICE[voice_lang]["start"])

    # 初始化摄像头
    if not camera.open():
        logger.error("摄像头初始化失败")
        return

    # 创建预览窗口
    win_name = zh2gbk("VisionLink Smart Glass")
    ui.create_window(win_name, CAMERA_CONFIG["width"], CAMERA_CONFIG["height"])

    logger.info("进入主循环")

    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        raw_frame = frame.copy()

        # 更新 UI 状态
        ui.update_state(infer._state if hasattr(infer, '_state') else 0,
                        current_mode, auto_enabled, voice_lang)
        ui_frame = ui.draw_panel(frame)

        now = time.time()

        # 自动模式
        if agent.should_scan(now):
            tasks = agent.detect_tasks(raw_frame)
            if tasks:
                agent.execute_tasks(tasks, raw_frame, FUNC_MAP)

        ui.show(win_name, ui_frame)

        # 无头模式兼容
        if not ui.enable_gui:
            time.sleep(0.02)
            continue

        key = cv2.waitKey(20) & 0xFF

        # S - 停止语音
        if key == ord('s') or key == ord('S'):
            tts.play_beep()
            tts.stop()
            continue

        # M - 切换自动模式
        if key == ord('m') or key == ord('M'):
            tts.play_beep()
            auto_enabled = agent.toggle()
            tip_key = "auto_on" if auto_enabled else "auto_off"
            tts.speak(TIP_VOICE[voice_lang][tip_key])
            continue

        # L - 切换语种
        if key == ord('l') or key == ord('L'):
            tts.play_beep()
            switch_language()
            continue

        # 1~5 - 切换模式
        if 49 <= key <= 53:
            tts.play_beep()
            current_mode = key - 48
            mode_txt = MODE_NAME_LIST[voice_lang][current_mode - 1]
            if voice_lang == "en":
                tts.speak(f"Switched to {mode_txt} mode")
            else:
                tts.speak(f"已切换至 {mode_txt} 模式")
            continue

        # 空格 - 执行（带防抖）
        if key == 32:
            if now - last_space_press < SPACE_DEBOUNCE:
                continue
            last_space_press = now
            tts.play_beep()
            func = FUNC_MAP.get(current_mode)
            if func:
                threading.Thread(target=func, args=(raw_frame.copy(),), daemon=True).start()
            continue

        # ESC - 退出
        if key == 27:
            tts.play_beep()
            logger.info("退出程序")
            tts.stop()
            tts.speak(TIP_VOICE[voice_lang]["exit"])
            break

    camera.release()
    ui.destroy()
    logger.info("程序结束")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"全局异常：{e}")
        raise
