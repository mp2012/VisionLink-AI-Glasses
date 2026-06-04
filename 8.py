import os
import sys
import time
import cv2
import ollama
import base64
import threading
import ctypes
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # 🔥 引入 Pillow 库支持中文字体渲染

sys.stdout.reconfigure(encoding='utf-8')

TARGET_MODEL = 'gemma4:e2b'
USB_CAMERA_ID = 1

PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
AI_IMAGE_SIZE = 448

current_mode = 1
mode_name = ["当前模式：避障模式 (Obstacle)", "当前模式：文字阅读 (OCR Text)", "当前模式：场景描述 (Scene)"]
is_ai_running = False

# 🔥 加载 Windows 系统自带的微软雅黑字体，字号设为 28
try:
    font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
    cv2_font_chinese = ImageFont.truetype(font_path, 28)
    cv2_font_small = ImageFont.truetype(font_path, 20)
except IOError:
    cv2_font_chinese = None


# 🔥 核心辅助函数：利用 PIL 在 OpenCV 矩阵上完美手写汉字
def draw_chinese_text(img, text, position, font, color=(0, 255, 0)):
    """ 完美的中文渲染转换管道 """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def play_shutter_sound():
    try:
        threading.Thread(target=lambda: ctypes.windll.winmm.PlaySoundW("SystemAsterisk", 0, 0x0001 | 0x00400000),
                         daemon=True).start()
    except BaseException:
        pass


def speak(text):
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
    global current_mode, is_ai_running
    is_ai_running = True
    try:
        h, w = frame.shape[:2]
        scale = AI_IMAGE_SIZE / w
        resized = cv2.resize(frame, (AI_IMAGE_SIZE, int(h * scale)))
        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        pure_b64 = base64.b64encode(buf).decode('utf-8')

        if current_mode == 1:
            prompt = "你是盲人眼镜的避障大脑。请仔细观察图片正中央，识别出最近的障碍物并用常识估算距离（限制在25字内，纯中文回答）。"
        elif current_mode == 2:
            prompt = "你是一款OCR文字阅读眼镜。请精确提取并辨认出这张图片里的所有中英文文字并直接朗读出来。"
        else:
            prompt = "你是一款导盲场景描述大脑。请仔细观察我眼前的画面，用温柔、充满常识的纯中文语言描述场景（50字内）。"

        res = ollama.chat(model=TARGET_MODEL, messages=[{'role': 'user', 'content': prompt, 'images': [pure_b64]}],
                          options={'temperature': 0.1})

        if 'message' in res and 'content' in res['message']:
            result = res['message']['content'].strip()
            if result: speak(result)
    except Exception as e:
        print(f"[DEV ERROR] ❌ 边缘端侧多模态计算链路崩溃: {e}")
        speak("识别失败")
    finally:
        is_ai_running = False


def main():
    global current_mode
    cap = cv2.VideoCapture(USB_CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return

    WINDOW_TITLE = "VisionLink AI"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    print("\n🚀 Google Hackathon demo 最终版启动成功！")
    sys.stdout.flush()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None: continue

        h, w, _ = frame.shape
        box_x1, box_y1 = int(w * 0.25), int(h * 0.2)
        box_x2, box_y2 = int(w * 0.75), int(h * 0.8)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

        # 🎨 背景牌绘制
        cv2.rectangle(frame, (20, 20), (560, 80), (0, 0, 0), -1)

        # 🔥 核心渲染：用 Pillow 替换传统的 putText，完美绘制中文
        if cv2_font_chinese:
            frame = draw_chinese_text(frame, mode_name[current_mode - 1], (35, 32), cv2_font_chinese, (0, 255, 0))
            if is_ai_running:
                cv2.rectangle(frame, (w - 280, h - 70), (w - 20, h - 20), (0, 0, 255), -1)
                frame = draw_chinese_text(frame, "AI 正在深度思考...", (w - 250, h - 58), cv2_font_small,
                                          (255, 255, 255))
            frame = draw_chinese_text(frame, "【空格】: 触发识别  |  【数字1/2/3】: 切换系统模式  |  【ESC】: 退出",
                                      (int(w * 0.1), h - 45), cv2_font_small, (255, 255, 255))
        else:
            # 兜底：如果找不到系统字体包，退回英文显示防炸
            cv2.putText(frame, f"MODE {current_mode}", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(15) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):
            if not is_ai_running:
                play_shutter_sound()
                snap = frame.copy()
                core_snap = snap[box_y1:box_y2, box_x1:box_x2]
                threading.Thread(target=analyze_frame, args=(core_snap,), daemon=True).start()
        elif key in [ord('1'), ord('2'), ord('3')]:
            current_mode = int(chr(key))
            speak(f"已切换{['避障模式', '文字阅读', '场景描述'][current_mode - 1]}")

        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()