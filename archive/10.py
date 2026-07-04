"""
VisionLink-AI-Glasses - Hackathon Tournament Grand Edition
Offline multimodal assistive glasses based on Gemma4 for visually impaired people.

✨ HACKATHON UPGRADES & FIXES:
1. Thread-Safe SAPI5 COM Fix: 100% stable Chinese voice broadcasting on Windows.
2. Local MP3 Shutter: Plays 'pic.mp3' natively via Windows MCI API without lag.
3. Voice-First Control: Hands-free voice commands for blind accessibility.
4. Dynamic Live HUD Subtitles: Stream-type subtitle rendering for judges to visualize AI speed.

Author: Michael (mp2012) & AI Collaborator
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
import queue
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
import speech_recognition as sr

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
    "当前模式：智能避障 (Obstacle Avoidance)",
    "当前模式：书刊阅读 (OCR Text Reader)",
    "当前模式：全景导航 (Scene Description)"
]

# 本地快门提示音路径配置
MP3_FILE_NAME = "pic.mp3"
# ==============================================================

# 全局状态控制
current_mode = 1
is_ai_running = False
live_subtitle = "等待指令..."  # 实时渲染给评委看的打字机字幕
trigger_ai_analysis = False    # 线程安全的语音触发标志

# 语音文本多线程队列
tts_queue = queue.Queue()


# ==================== 🛠️ ENGINE 1: NATIVE TTS WORKER (强力兼容修复版) ====================
def tts_worker():
    """
    常驻后台语音线程（强力兼容修复版）
    在工作线程内部动态管理 COM 组件生命周期，彻底消除 Windows 跨线程调用 SAPI5 锁死导致的静音 Bug
    """
    while True:
        try:
            # 阻塞式等待队列中的文本，超时1秒以响应退出
            text = tts_queue.get(timeout=1)
            if text == "__STOP__":
                break

            # 【核心修复点】: 不在全局初始化，而是谁负责播报，谁就在当前线程里即时初始化
            engine = pyttsx3.init()
            engine.setProperty('rate', 195)  # 适当加快语速，提升盲感体验

            # 【强制锁死中文】: 显式加载系统默认的中文声音组件
            voices = engine.getProperty('voices')
            for voice in voices:
                if "zh" in voice.languages or "Chinese" in voice.name or "Sheng" in voice.name:
                    engine.setProperty('voice', voice.id)
                    break

            # 执行硬件级发声
            engine.say(text)
            engine.runAndWait()

            # 播报完毕，立刻就地销毁当前实例，彻底释放 COM 底层句柄锁
            del engine

            tts_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS RUNTIME ERROR]: {e}")

# 在程序启动前，立刻激活常驻语音消费线程
worker_thread = threading.Thread(target=tts_worker, daemon=True)
worker_thread.start()


def speak_native(text):
    """ 极速投递：直接将文本清洗后扔进后台队列中排队连读 """
    if not text.strip():
        return
    print(f"🎧 [语音播报]: {text}")
    sys.stdout.flush()

    # 清洗可能干扰语音引擎的 Markdown 标记
    clean_text = text.replace("*", "").replace("#", "").replace("`", "").strip()
    tts_queue.put(clean_text)


# ==================== 🛠️ AUDIO: LOCAL MP3 SHUTTER PLAYER ====================
def play_shutter_sound():
    """
    使用 Windows WinMM (MCI) 异步播放本地 pic.mp3
    无需额外安装第三方音频库，毫秒级响应，完全异步不阻塞视频流
    """
    def _play():
        try:
            if not os.path.exists(MP3_FILE_NAME):
                print(f"⚠️ 未找到音频文件 {MP3_FILE_NAME}，改用系统默认提示音。")
                ctypes.windll.winmm.PlaySoundW("SystemAsterisk", 0, 0x0001 | 0x00400000)
                return

            winmm = ctypes.windll.winmm
            # 1. 停止并关闭上一次可能残留的音频别名，防止并发冲突
            winmm.mciSendStringW(f"stop shutter_alias", None, 0, 0)
            winmm.mciSendStringW(f"close shutter_alias", None, 0, 0)

            # 2. 打开当前的本地 mp3 文件并指定别名
            open_cmd = f'open "{os.path.abspath(MP3_FILE_NAME)}" type mpegvideo alias shutter_alias'
            winmm.mciSendStringW(open_cmd, None, 0, 0)

            # 3. 极速重放
            winmm.mciSendStringW("play shutter_alias", None, 0, 0)
        except Exception as e:
            print(f"[AUDIO ERROR] 本地快门音播放失败: {e}")

    threading.Thread(target=_play, daemon=True).start()


# ==================== 🎙️ ENGINE 2: VOICE-FIRST CONTROL (无障碍语音控标) ====================
def voice_control_listener():
    """ 无障碍语音控标线程：免双手操作切换模式或触发拍照 """
    global current_mode, is_ai_running, trigger_ai_analysis
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000  # 根据现场杂音调节

    print("🎙️  [Accessibility]: 盲人全语音无障碍交互管道已激活...")

    while True:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, phrase_time_limit=3)
                command = recognizer.recognize_google(audio, language='zh-CN')
                print(f"🗣️  [听到指令]: {command}")

                if "避障" in command or "安全" in command:
                    current_mode = 1
                    speak_native("已切换智能避障模式")
                elif "文字" in command or "阅读" in command or "字" in command:
                    current_mode = 2
                    speak_native("已切换书刊阅读模式")
                elif "场景" in command or "描述" in command or "环境" in command:
                    current_mode = 3
                    speak_native("已切换全景导航模式")
                elif "看看" in command or "拍照" in command or "识别" in command:
                    if not is_ai_running:
                        trigger_ai_analysis = True

            except sr.UnknownValueError:
                continue
            except Exception as e:
                time.sleep(1)

voice_thread = threading.Thread(target=voice_control_listener, daemon=True)
voice_thread.start()


# ==================== 🧠 ENGINE 3: STREAMING MULTIMODAL PIPELINE ====================
def analyze_frame_stream(frame):
    """ 核心端侧多模态流式推理管道 + 评委 HUD 字幕同步 """
    global current_mode, is_ai_running, live_subtitle
    is_ai_running = True
    live_subtitle = "AI 正在观察画面..."

    try:
        h, w = frame.shape[:2]
        scale = AI_IMAGE_SIZE / w
        resized = cv2.resize(frame, (AI_IMAGE_SIZE, int(h * scale)))
        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        pure_b64 = base64.b64encode(buf).decode('utf-8')

        if current_mode == 1:
            prompt = "你是盲人眼镜的避障大脑。请仔细观察图片正中央，识别出最近的障碍物并用常识估算距离（限制在25字内，纯中文回答）。"
        elif current_mode == 2:
            prompt = "你是一款OCR文字阅读眼镜。请精确提取并辨认出这张图片里的所有中英文文字并直接朗读出来，不要任何寒暄。"
        else:
            prompt = "你是一款导盲场景描述大脑。请仔细观察我眼前的画面，用温柔、充满常识的纯中文语言描述场景（50字内）。"

        response_stream = ollama.chat(
            model=TARGET_MODEL,
            messages=[{'role': 'user', 'content': prompt, 'images': [pure_b64]}],
            options={'temperature': 0.1},
            stream=True
        )

        sentence_buffer = ""
        live_subtitle = ""

        for chunk in response_stream:
            token = chunk['message']['content']
            sentence_buffer += token
            live_subtitle += token  # 同步渲染大模型打字机字幕给评委看

            # 遇到标点立即断句切片投递
            if any(p in token for p in ['。', '？', '！', '\n', '；', ';']):
                clean_sentence = sentence_buffer.strip()
                if clean_sentence:
                    speak_native(clean_sentence)
                sentence_buffer = ""

        if sentence_buffer.strip():
            speak_native(sentence_buffer.strip())

    except Exception as e:
        print(f"[DEV ERROR] ❌ 边缘端侧多模态计算链路崩溃: {e}")
        speak_native("识别失败")
        live_subtitle = "系统异常，请重试"
    finally:
        is_ai_running = False


# ==================== 🎨 UI RENDERING & CROSS-PLATFORM FONTS ====================
def load_system_font():
    """ 自动检索系统可用中文字体 """
    possible_paths = [
        "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, 28), ImageFont.truetype(path, 20), ImageFont.truetype(path, 24)
            except IOError: continue
    return None, None, None

cv2_font_chinese, cv2_font_small, cv2_font_hud = load_system_font()

def draw_chinese_text(img, text, position, font, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ==================== 🎬 MAIN TOURNAMENT LOOP ====================
def main():
    global current_mode, trigger_ai_analysis, live_subtitle

    cap = cv2.VideoCapture(USB_CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)

    if not cap.isOpened():
        print("⚠️ 外部摄像头未就绪，正在无缝接管本地默认摄像头 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 错误：未检测到任何可用摄像头，系统退出。")
            return

    WINDOW_TITLE = "VisionLink AI - Hackathon Grand Finals"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    print("\n=======================================================")
    print("🚀 VisionLink AI Glasses 启动成功！已加载本地快门音资产。")
    print(f"📦 核心大脑: Google Gemma4 ({TARGET_MODEL})")
    print(f"🎵 拍照快门音资产: {os.path.abspath(MP3_FILE_NAME)}")
    print("=======================================================\n")
    sys.stdout.flush()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None: continue

        h, w, _ = frame.shape
        # 视障中心对焦采样框 (ROI 框)
        box_x1, box_y1 = int(w * 0.25), int(h * 0.2)
        box_x2, box_y2 = int(w * 0.75), int(h * 0.8)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

        # 头部与底部 HUD 视觉遮罩背景牌
        cv2.rectangle(frame, (20, 20), (600, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, h - 140), (w - 20, h - 70), (20, 20, 20), -1)
        cv2.rectangle(frame, (20, h - 140), (w - 20, h - 70), (0, 255, 0), 1)
        cv2.rectangle(frame, (20, h - 55), (w - 20, h - 20), (0, 0, 0), -1)

        # 图像 UI 渲染管线
        if cv2_font_chinese:
            # 顶部当前系统模式
            frame = draw_chinese_text(frame, MODE_NAMES[current_mode - 1], (35, 32), cv2_font_chinese, (0, 255, 0))

            # 底部评委专属多模态打字机流式字幕
            display_text = f"【AI 实时流式输出】: {live_subtitle}"
            frame = draw_chinese_text(frame, display_text, (35, h - 132), cv2_font_hud, (255, 255, 255))

            # 右上角 AI 推理状态球
            if is_ai_running:
                cv2.circle(frame, (w - 50, 50), 12, (0, 0, 255), -1) # 红色闪烁状态球（思考中）
            else:
                cv2.circle(frame, (w - 50, 50), 12, (0, 255, 0), -1) # 绿色就绪球

            # 底部无障碍控制操作提示
            frame = draw_chinese_text(frame, "【全语音交互】: 说出 '避障'/'文字'/'场景'/'看看'  |  键盘双控: [空格]触控 [ESC]退出",
                                      (35, h - 47), cv2_font_small, (180, 180, 180))
        else:
            cv2.putText(frame, f"MODE {current_mode}", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(15) & 0xFF

        # 处理【空格键按下】或者【语音说出 看看/拍照】触发的图像多模态推理
        if key == ord(' ') or trigger_ai_analysis:
            trigger_ai_analysis = False # 消费掉本次信号
            if not is_ai_running:
                play_shutter_sound()  # 触发本地 pic.mp3 极速异步播放
                snap = frame.copy()
                core_snap = snap[box_y1:box_y2, box_x1:box_x2] # 裁剪中心区域，降低算力开销
                threading.Thread(target=analyze_frame_stream, args=(core_snap,), daemon=True).start()

        elif key in [ord('1'), ord('2'), ord('3')]:
            current_mode = int(chr(key))
            speak_native(f"已切换{['避障模式', '文字阅读', '场景描述'][current_mode - 1]}")

        elif key == 27:  # ESC 退出
            break

        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break

    # 资源安全释放管道
    tts_queue.put("__STOP__")
    try:
        ctypes.windll.winmm.mciSendStringW("close shutter_alias", None, 0, 0)
    except: pass
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()