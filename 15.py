"""
VisionLink-AI-Glasses - Main Entry Point
Cross-platform multimodal assistive glasses based on Gemma4/Ollama.
Fully optimized for Windows (Dev) & Jetson Orin Nano (Edge Deploy).
Fix: Slow Gemma4:e2b inference timeout deadlock
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
import platform
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 强制标准输出采用 UTF-8 编码
sys.stdout.reconfigure(encoding='utf-8')

# ==================== CROSS-PLATFORM DETECT ====================
IS_JETSON = (platform.system() == "Linux" and platform.machine() == "aarch64")
# 硬编码关闭图形界面，无头部署
show_window = False
print("💡 已硬编码禁用所有OpenCV图形窗口，无头模式运行")
# ==============================================================

# ==================== GLOBAL CONFIGURATION ====================
TARGET_MODEL = 'gemma4:e2b'
# 适配慢速大模型：缩小图片降低推理耗时
AI_IMAGE_SIZE = 384
JPEG_QUALITY = 70
# 自动扫描间隔：6秒，给大模型充足推理时间
SCAN_INTERVAL = 6
# 单轮推理最大超时阈值（超过则强制释放锁）
INFER_TIMEOUT = 8

if IS_JETSON:
    USB_CAMERA_ID = 0
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 480
else:
    USB_CAMERA_ID = 1
    PREVIEW_WIDTH = 1280
    PREVIEW_HEIGHT = 720

MODE_NAMES = [
    "当前模式：避障模式 (Obstacle)",
    "当前模式：文字阅读 (OCR Text)",
    "当前模式：场景描述 (Scene)"
]
# ==============================================================

current_mode = 1
is_ai_running = False
last_auto_scan_ts = 0


def load_system_font():
    possible_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                font_large = ImageFont.truetype(path, 24 if IS_JETSON else 28)
                font_small = ImageFont.truetype(path, 16 if IS_JETSON else 20)
                return font_large, font_small
            except IOError:
                continue
    return None, None


cv2_font_chinese, cv2_font_small = load_system_font()


def draw_chinese_text(img, text, position, font, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def play_shutter_sound():
    def play():
        if IS_JETSON:
            os.system("(speaker-test -t sine -f 1000 -l 1 &) && sleep 0.1 && killall speaker-test > /dev/null 2>&1")
        else:
            import ctypes
            try:
                ctypes.windll.winmm.PlaySoundW("SystemAsterisk", 0, 0x0001 | 0x00400000)
            except BaseException:
                pass
    threading.Thread(target=play, daemon=True).start()


def speak(text):
    print(f"🎧 [语音播报]: {text}")
    sys.stdout.flush()
    clean_text = text.replace('"', '').replace("'", "").replace('“', '').replace('”', '').replace("\n", " ")
    def run_cmd():
        if IS_JETSON:
            os.system(f"espeak -v zh+f2 -s 160 '{clean_text}' > /dev/null 2>&1")
        else:
            cmd = f'''powershell -c "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{clean_text}');"'''
            os.system(cmd)
    threading.Thread(target=run_cmd, daemon=True).start()


def analyze_frame(frame):
    global current_mode, is_ai_running
    is_ai_running = True

    # 子线程封装真实推理逻辑
    def inference_task():
        try:
            h, w = frame.shape[:2]
            scale = AI_IMAGE_SIZE / w
            resized = cv2.resize(frame, (AI_IMAGE_SIZE, int(h * scale)))
            _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            b64 = base64.b64encode(buf).decode('utf-8')

            if current_mode == 1:
                prompt = "你是盲人眼镜的避障大脑。请仔细观察图片正中央，识别出最近的障碍物并用常识估算距离（限制在25字内，纯中文回答）。"
            elif current_mode == 2:
                prompt = "你是一款OCR文字阅读眼镜。请精确提取并辨认出这张图片里的所有中英文文字并直接朗读出来。"
            else:
                prompt = "你是一款导盲场景描述大脑。请仔细观察我眼前的画面，用温柔、充满常识的纯中文语言描述场景（50字内）。"

            res = ollama.chat(
                model=TARGET_MODEL,
                messages=[{"role": "user", "content": prompt, "images": [b64]}],
                options={"temperature": 0.1}
            )
            if res.get("message", {}).get("content"):
                speak(res["message"]["content"].strip())
        except Exception as e:
            print(f"[ERROR] AI推理异常: {e}")
            speak("识别超时失败")

    # 启动推理线程并设置超时等待
    infer_thread = threading.Thread(target=inference_task, daemon=True)
    infer_thread.start()
    # 最多等待INFER_TIMEOUT秒，无论是否跑完都释放锁
    infer_thread.join(timeout=INFER_TIMEOUT)
    is_ai_running = False
    print(f"⏱️ 本轮推理流程结束，已释放运行锁")


def main():
    global current_mode, last_auto_scan_ts
    # 摄像头初始化
    if IS_JETSON:
        cap = cv2.VideoCapture(USB_CAMERA_ID, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(USB_CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)

    if not cap.isOpened():
        print(f"⚠️ 摄像头{USB_CAMERA_ID}打开失败，尝试0号摄像头")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 无可用摄像头，退出")
            return

    WINDOW_TITLE = "VisionLink AI - On Device Assistant"
    if show_window:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    print("\n=======================================================")
    print(f"🚀 运行平台: Jetson Orin Nano")
    print(f"📦 模型: {TARGET_MODEL}")
    print(f"🖥️ 图形界面: 关闭(无头自动扫描)")
    print(f"⚙️ 扫描间隔: {SCAN_INTERVAL}秒 | 推理最大超时: {INFER_TIMEOUT}秒")
    print(f"⚙️ AI输入图尺寸: {AI_IMAGE_SIZE}")
    print("=======================================================\n")
    sys.stdout.flush()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            h, w = frame.shape[:2]
            x1, y1 = int(w*0.25), int(h*0.2)
            x2, y2 = int(w*0.75), int(h*0.8)
            roi = frame[y1:y2, x1:x2]

            # 无头自动扫描逻辑
            now = time.time()
            if now - last_auto_scan_ts >= SCAN_INTERVAL and not is_ai_running:
                last_auto_scan_ts = now
                print(f"🤖 触发自动视觉扫描（间隔{SCAN_INTERVAL}s）")
                threading.Thread(target=analyze_frame, args=(roi,), daemon=True).start()

            # 主线程轻微休眠，降低CPU占用，不阻塞摄像头采集
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[INFO] 收到Ctrl+C中断，准备退出")
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        print("[INFO] 摄像头资源释放完成，程序退出")


if __name__ == "__main__":
    main()
