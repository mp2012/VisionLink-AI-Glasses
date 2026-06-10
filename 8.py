"""
VisionLink-AI-Glasses - Main Entry Point
Offline multimodal assistive glasses based on Gemma4 for visually impaired people.

Author: Michael (mp2012)
Date: 2026-06
License: MIT
"""

import os
import sys
import time
import cv2
import ollama
import base64
import threading
import ctypes
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 强制标准输出采用 UTF-8 编码，防止 Windows 控制台打印中文乱码
sys.stdout.reconfigure(encoding='utf-8')

# ==================== GLOBAL CONFIGURATION ====================
TARGET_MODEL = 'gemma4:e2b'
USB_CAMERA_ID = 1

# 视频流与边缘端推理分辨率配置
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
AI_IMAGE_SIZE = 448

# 系统模式映射表
MODE_NAMES = [
    "当前模式：避障模式 (Obstacle)",
    "当前模式：文字阅读 (OCR Text)",
    "当前模式：场景描述 (Scene)"
]
# ==============================================================

# 全局状态变量
current_mode = 1
is_ai_running = False


def load_system_font():
    """ 跨平台自动检索系统可用中文字体，提升 Demo 的可移植性 """
    possible_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
        "/System/Library/Fonts/PingFang.ttc",  # macOS 东方青龙（平方）
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # Linux 常用文泉驿
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                font_large = ImageFont.truetype(path, 28)
                font_small = ImageFont.truetype(path, 20)
                return font_large, font_small
            except IOError:
                continue
    return None, None


# 初始化全局字体
cv2_font_chinese, cv2_font_small = load_system_font()


def draw_chinese_text(img, text, position, font, color=(0, 255, 0)):
    """ 利用 PIL 在 OpenCV 图像矩阵上无损渲染汉字 """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def play_shutter_sound():
    """ 异步触发 Windows 系统内置快门提示音 """
    try:
        # SND_ALIAS (0x0001) | SND_ASYNC (0x00400000)
        threading.Thread(
            target=lambda: ctypes.windll.winmm.PlaySoundW("SystemAsterisk", 0, 0x0001 | 0x00400000),
            daemon=True
        ).start()
    except BaseException:
        pass


def speak(text):
    """ 基于 PowerShell 异步调用 Windows 内置 TTS 引擎，确保不卡顿视频流 """
    print(f"🎧 [语音播报]: {text}")
    sys.stdout.flush()
    try:
        clean_text = text.replace('"', '').replace("'", "").replace('“', '').replace('”', '').replace("\n", " ")
        command = f'''powershell -c "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{clean_text}');"'''

        def run_cmd():
            os.system(command)

        threading.Thread(target=run_cmd, daemon=True).start()
    except Exception as e:
        print(f"[DEBUG AUDIO ERROR] 语音模块异常: {e}")


def analyze_frame(frame):
    """ 核心端侧多模态推理管道 """
    global current_mode, is_ai_running
    is_ai_running = True
    try:
        h, w = frame.shape[:2]
        scale = AI_IMAGE_SIZE / w
        resized = cv2.resize(frame, (AI_IMAGE_SIZE, int(h * scale)))
        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        pure_b64 = base64.b64encode(buf).decode('utf-8')

        # 根据当前模式自动切换多模态 Prompt 策略
        if current_mode == 1:
            prompt = "你是盲人眼镜的避障大脑。请仔细观察图片正中央，识别出最近的障碍物并用常识估算距离（限制在25字内，纯中文回答）。"
        elif current_mode == 2:
            prompt = "你是一款OCR文字阅读眼镜。请精确提取并辨认出这张图片里的所有中英文文字并直接朗读出来。"
        else:
            prompt = "你是一款导盲场景描述大脑。请仔细观察我眼前的画面，用温柔、充满常识的纯中文语言描述场景（50字内）。"

        # 发起端侧大模型多模态调用
        res = ollama.chat(
            model=TARGET_MODEL,
            messages=[{'role': 'user', 'content': prompt, 'images': [pure_b64]}],
            options={'temperature': 0.1}
        )

        if 'message' in res and 'content' in res['message']:
            result = res['message']['content'].strip()
            if result:
                speak(result)
    except Exception as e:
        print(f"[DEV ERROR] ❌ 边缘端侧多模态计算链路崩溃: {e}")
        speak("识别失败")
    finally:
        is_ai_running = False


def main():
    global current_mode

    # 初始化摄像头驱动
    cap = cv2.VideoCapture(USB_CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)

    # 兜底机制：若指定 USB 外部摄像头不可用，自动切回主摄像头
    if not cap.isOpened():
        print(f"⚠️ 外部摄像头ID {USB_CAMERA_ID} 无法打开，尝试接管本地默认摄像头 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 错误：未检测到任何可用摄像头，系统退出。")
            return

    WINDOW_TITLE = "VisionLink AI - On Device Assistant"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    print("\n=======================================================")
    print("🚀 Google Hackathon - VisionLink AI Glasses 启动成功！")
    print(f"📦 当前加载端侧模型: {TARGET_MODEL}")
    print("=======================================================\n")
    sys.stdout.flush()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w, _ = frame.shape
        # 视障中心对焦采样框 (ROI 框)
        box_x1, box_y1 = int(w * 0.25), int(h * 0.2)
        box_x2, box_y2 = int(w * 0.75), int(h * 0.8)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

        # 头部 HUD UI 信息遮罩背景牌
        cv2.rectangle(frame, (20, 20), (560, 80), (0, 0, 0), -1)

        # 图像 UI 渲染管线
        if cv2_font_chinese:
            # 顶部模式状态渲染
            frame = draw_chinese_text(frame, MODE_NAMES[current_mode - 1], (35, 32), cv2_font_chinese, (0, 255, 0))

            # 推理状态实时气泡
            if is_ai_running:
                cv2.rectangle(frame, (w - 280, h - 70), (w - 20, h - 20), (0, 0, 255), -1)
                frame = draw_chinese_text(frame, "AI 正在深度思考...", (w - 250, h - 58), cv2_font_small,
                                          (255, 255, 255))

            # 底部快捷键状态栏
            frame = draw_chinese_text(frame, "【空格】: 触发识别  |  【数字1/2/3】: 切换系统模式  |  【ESC】: 退出",
                                      (int(w * 0.1), h - 45), cv2_font_small, (255, 255, 255))
        else:
            # 极端情况无系统字体资产时的 ASCII 码兜底渲染
            cv2.putText(frame, f"MODE {current_mode}", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(WINDOW_TITLE, frame)

        # 键盘中断流控制
        key = cv2.waitKey(15) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # 空格触发图像捕获与深度多模态分析
            if not is_ai_running:
                play_shutter_sound()
                snap = frame.copy()
                core_snap = snap[box_y1:box_y2, box_x1:box_x2]  # 仅裁剪中心 ROI 发送给大模型，节省端侧算力
                threading.Thread(target=analyze_frame, args=(core_snap,), daemon=True).start()
        elif key in [ord('1'), ord('2'), ord('3')]:  # 切换产品模式
            current_mode = int(chr(key))
            speak(f"已切换{['避障模式', '文字阅读', '场景描述'][current_mode - 1]}")

        # 用户点击窗口关闭 [X] 时安全退出
        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break

    # 资源安全释放管道
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()