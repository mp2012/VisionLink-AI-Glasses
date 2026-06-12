"""
VisionLink-AI-Glasses
强制调用外置USB摄像头(索引1)，屏蔽笔记本内置摄像头(索引0)
修复：Ctrl+Q按键误触发、摄像头兼容性、资源释放、global语法错误
按键：空格=识别/打断任务 | Ctrl+Q=中断语音 | 1-5切换模式 | L=语种 | ESC=退出
"""
import os
import sys
import cv2
import ollama
import base64
import threading
import ctypes
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
import traceback
import logging
import speech_recognition as sr
import time

# ==================== 解决 OpenCV 窗口中文乱码 ====================
def zh2gbk(s):
    return s.encode("gbk", errors="replace").decode("gbk")
# =================================================================

# ==================== 日志初始化 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(encoding='utf-8')

# ==================== 全局基础配置 ====================
TARGET_MODEL = 'gemma4:e2b'
# 强制使用 外置USB摄像头索引 1，内置0直接跳过
USB_CAM_ID = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
AI_IMAGE_SIZE = 448
FRAME_CROP_RATIO = 0.25

# 模式文案
MODE_TIPS = {
    "zh": ["避障模式", "文字阅读模式", "场景描述模式", "人脸识别模式", "跨境对话模式"],
    "en": ["Obstacle mode", "Text reading mode", "Scene description mode", "Face recognition mode", "Cross-border dialogue mode"]
}
LANG_TIPS = {
    "zh": "语言已切换",
    "en": "Language switched"
}
VOICE_TIPS = {
    "zh": {
        "system_start": "系统启动成功",
        "busy": "正在识别，请稍等",
        "capture": "开始识别",
        "recognize_fail": "识别失败",
        "exit": "程序退出",
        "face_detected": "检测到人脸",
        "no_face": "未检测到人脸",
        "listen_start": "开始聆听，请说话",
        "listen_fail": "未听清，请重新说",
        "stop_task": "已停止"
    },
    "en": {
        "system_start": "System started",
        "busy": "Processing, please wait",
        "capture": "Start recognition",
        "recognize_fail": "Recognition failed",
        "exit": "Program exit",
        "face_detected": "Face detected",
        "no_face": "No face detected",
        "listen_start": "Listening...",
        "listen_fail": "Failed to recognize, please try again",
        "stop_task": "Stopped"
    }
}
TARGET_LANG = 'zh'
# ==========================================================

current_mode = 1
ai_running_event = threading.Event()
listen_running = threading.Event()
GLOBAL_STOP_FLAG = threading.Event()
GLOBAL_TTS_LOCK = threading.Lock()

# 全局TTS引擎（用于实时停止语音）
tts_engine = pyttsx3.init()

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
last_face_state = None

# ==================== TTS 语音播报 ====================
def tts_speak(text, lang="zh"):
    if not text or GLOBAL_STOP_FLAG.is_set():
        return
    def _run():
        with GLOBAL_TTS_LOCK:
            try:
                logger.info(f"语音播报: {text}")
                tts_engine.setProperty('rate', 180)
                tts_engine.setProperty('volume', 1.0)
                voices = tts_engine.getProperty('voices')

                for v in voices:
                    name = v.name.lower()
                    vid = v.id.lower()
                    if lang == "zh":
                        if "chinese" in name or "zh" in vid:
                            tts_engine.setProperty('voice', v.id)
                            break
                    else:
                        if "english" in name or "en" in vid:
                            tts_engine.setProperty('voice', v.id)
                            break

                tts_engine.say(text)
                tts_engine.runAndWait()
                logger.info("播报完成")
            except Exception as e:
                logger.error(f"语音异常: {str(e)}")
    threading.Thread(target=_run, daemon=True).start()

# ==================== 字体加载与绘制 ====================
def load_system_font():
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc"
    ]
    font_lg, font_sm = None, None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font_lg = ImageFont.truetype(path, 24)
                font_sm = ImageFont.truetype(path, 18)
                logger.info(f"加载字体成功: {path}")
                break
            except:
                logger.warning(f"字体 {path} 加载失败")
    if not font_lg:
        font_lg = ImageFont.load_default(size=24)
    if not font_sm:
        font_sm = ImageFont.load_default(size=18)
    return font_lg, font_sm

font_lg, font_sm = load_system_font()

def draw_text(img, text, pos, font, color=(0, 255, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 快门音效
def shutter():
    try:
        threading.Thread(
            target=lambda: ctypes.windll.winmm.PlaySoundW("SystemAsterisk", 0, 1 | 0x400000),
            daemon=True
        ).start()
    except Exception:
        logger.warning("快门音效播放失败")

# ==================== 通用AI图像分析 ====================
def analyze_frame(frame, mode, lang):
    global ai_running_event
    try:
        if GLOBAL_STOP_FLAG.is_set():
            return
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            tts_speak(VOICE_TIPS[lang]["recognize_fail"], lang)
            return

        scale = AI_IMAGE_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]
        ret, buf = cv2.imencode(".jpg", resized, encode_param)
        if not ret or GLOBAL_STOP_FLAG.is_set():
            tts_speak(VOICE_TIPS[lang]["recognize_fail"], lang)
            return
        b64_data = base64.b64encode(buf).decode()

        if mode == 1:
            prompt = "观察画面，判断前方障碍物，简短口语提醒。"
        elif mode == 2:
            prompt = "提取图片中所有文字，只输出文字内容。"
        elif mode == 3:
            prompt = "简洁描述当前画面场景，口语化表达。"
        else:
            prompt = ""

        resp = ollama.chat(
            model=TARGET_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [b64_data]}],
            options={"temperature": 0.1}
        )
        if GLOBAL_STOP_FLAG.is_set():
            return
        result_text = resp["message"]["content"].strip()
        logger.info(f"AI返回: {result_text}")
        if result_text:
            tts_speak(result_text, lang)
        else:
            tts_speak(VOICE_TIPS[lang]["recognize_fail"], lang)
    except Exception as e:
        logger.error(f"AI异常:\n{traceback.format_exc()}")
        tts_speak(VOICE_TIPS[lang]["recognize_fail"], lang)
    finally:
        ai_running_event.clear()

# ==================== 人脸检测逻辑 ====================
def face_detect_process(frame, lang):
    global last_face_state
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    current_state = "face" if len(faces) > 0 else "none"
    if current_state != last_face_state:
        last_face_state = current_state
        if current_state == "face":
            tts_speak(VOICE_TIPS[lang]["face_detected"], lang)
        else:
            tts_speak(VOICE_TIPS[lang]["no_face"], lang)
    return frame

# ==================== 跨境对话翻译（模式5） ====================
def cross_dialog_worker(frame, lang):
    global listen_running
    listen_running.set()
    r = sr.Recognizer()
    r.energy_threshold = 3000
    r.dynamic_energy_threshold = True

    h, w = frame.shape[:2]
    scale = AI_IMAGE_SIZE / max(w, h)
    resized = cv2.resize(frame, (int(w*scale), int(h*scale)))
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()

    try:
        if GLOBAL_STOP_FLAG.is_set():
            return
        with sr.Microphone() as source:
            logger.info("开始收音...")
            tts_speak(VOICE_TIPS[lang]["listen_start"], lang)
            audio = r.listen(source, timeout=5, phrase_time_limit=8)

        if GLOBAL_STOP_FLAG.is_set():
            return
        try:
            text = r.recognize_google(audio, language="zh-CN,en-US")
            logger.info(f"识别语音内容: {text}")
        except:
            tts_speak(VOICE_TIPS[lang]["listen_fail"], lang)
            return

        if GLOBAL_STOP_FLAG.is_set():
            return
        prompt = f"""
你是出国旅游随身翻译助手，结合眼前画面和对方语音内容完成工作：
1. 自动识别语种，外语翻译成自然中文，中文翻译成地道外语；
2. 结合画面里的物品、手势、场景，补充关键提醒（价格、禁忌、安全、过敏等）；
3. 如需交流，生成简短、礼貌、符合当地习惯的口语回复；
4. 全程短句，通俗易懂，适合语音朗读。
当前语音内容：{text}
        """
        resp = ollama.chat(
            model=TARGET_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
            options={"temperature": 0.2}
        )
        if GLOBAL_STOP_FLAG.is_set():
            return
        reply = resp["message"]["content"].strip()
        logger.info(f"Gemma4翻译&回复: {reply}")
        tts_speak(reply, lang)

    except Exception as e:
        logger.error(f"跨境对话异常: {str(e)}")
        tts_speak(VOICE_TIPS[lang]["recognize_fail"], lang)
    finally:
        listen_running.clear()

# ==================== 主函数 ====================
def main():
    global current_mode, ai_running_event, TARGET_LANG, last_face_state, listen_running, GLOBAL_STOP_FLAG
    logger.info("===== VisionLink AI眼镜 启动 =====")

    # 打开索引1 = 外置USB摄像头，兼容多后端
    cap = cv2.VideoCapture(USB_CAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(USB_CAM_ID)
        logger.warning("CAP_DSHOW 打开失败，切换默认摄像头后端")

    if not cap.isOpened():
        logger.error("无法打开外置USB摄像头，请检查USB接线、设备是否通电！")
        return

    # 等待USB设备初始化
    time.sleep(0.5)

    # 设置分辨率并读取实际生效值
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"外置USB摄像头已就绪，分辨率: {real_w} x {real_h}")

    win_title = "VisionLink AI眼镜"
    cv2.namedWindow(zh2gbk(win_title), cv2.WINDOW_NORMAL)
    tts_speak(VOICE_TIPS[TARGET_LANG]["system_start"], TARGET_LANG)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("读取摄像头画面失败")
            continue
        frame_copy = frame.copy()
        h, w = frame.shape[:2]

        # 重置打断标记
        GLOBAL_STOP_FLAG.clear()

        # 文字模式选区框
        crop_pad = int(min(w, h) * FRAME_CROP_RATIO)
        bx1, by1 = crop_pad, crop_pad
        bx2, by2 = w - crop_pad, h - crop_pad
        if current_mode == 2:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

        # 人脸识别
        if current_mode == 4:
            frame = face_detect_process(frame, TARGET_LANG)

        # 顶部状态栏
        cv2.rectangle(frame, (20, 20), (600, 80), (0, 0, 0), -1)
        status_text = f"模式：{MODE_TIPS[TARGET_LANG][current_mode-1]} | 语言：{TARGET_LANG.upper()}"
        frame = draw_text(frame, status_text, (35, 35), font_lg)

        # 状态提示
        if ai_running_event.is_set():
            cv2.rectangle(frame, (w-220, h-60), (w-20, h-20), (0, 0, 255), -1)
            frame = draw_text(frame, "识别中...", (w-210, h-40), font_sm, (255,255,255))
        if listen_running.is_set():
            cv2.rectangle(frame, (w-220, h-60), (w-20, h-20), (255, 100, 0), -1)
            frame = draw_text(frame, "聆听中...", (w-210, h-40), font_sm, (255,255,255))

        # 底部按键提示
        tip_text = "空格:识别/打断任务 | Ctrl+Q:中断语音 | 1-5:切换模式 | L:语言 | ESC:退出"
        frame = draw_text(frame, tip_text, (10, h-20), font_sm, (255,255,255))

        cv2.imshow(zh2gbk(win_title), frame)
        key = cv2.waitKey(15)

        # --------------- Ctrl+Q 中断语音 ---------------
        if (key & 0xFF) == ord('q') and (key & 0xFF00) != 0:
            with GLOBAL_TTS_LOCK:
                tts_engine.stop()
            logger.info("已中断语音播报")
            continue

        key_low = key & 0xFF
        # 空格：识别 / 打断AI/收音任务
        if key_low == ord(' '):
            if ai_running_event.is_set() or listen_running.is_set():
                GLOBAL_STOP_FLAG.set()
                ai_running_event.clear()
                listen_running.clear()
                tts_speak(VOICE_TIPS[TARGET_LANG]["stop_task"], TARGET_LANG)
                logger.info("手动打断当前任务")
                continue
            # 空闲状态启动识别
            ai_running_event.set()
            shutter()
            tts_speak(VOICE_TIPS[TARGET_LANG]["capture"], TARGET_LANG)
            if current_mode == 2:
                snap = frame_copy[by1:by2, bx1:bx2].copy()
            else:
                snap = frame_copy.copy()
            threading.Thread(target=analyze_frame, args=(snap, current_mode, TARGET_LANG), daemon=True).start()

        # 切换模式 1/2/3/4/5
        elif key_low in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            if ai_running_event.is_set() or listen_running.is_set():
                GLOBAL_STOP_FLAG.set()
                ai_running_event.clear()
                listen_running.clear()
                tts_speak(VOICE_TIPS[TARGET_LANG]["stop_task"], TARGET_LANG)
            new_mode = int(chr(key_low))
            if new_mode != current_mode:
                current_mode = new_mode
                mode_name = MODE_TIPS[TARGET_LANG][current_mode-1]
                logger.info(f"切换至 {mode_name}")
                if current_mode == 4:
                    last_face_state = None
                tts_speak(mode_name, TARGET_LANG)

        # L 切换语种
        elif key_low in (ord('l'), ord('L')):
            TARGET_LANG = "en" if TARGET_LANG == "zh" else "zh"
            tts_speak(LANG_TIPS[TARGET_LANG], TARGET_LANG)

        # ESC 打断全部并退出
        elif key_low == 27:
            GLOBAL_STOP_FLAG.set()
            ai_running_event.clear()
            listen_running.clear()
            with GLOBAL_TTS_LOCK:
                tts_engine.stop()
            logger.info("按下ESC，退出程序")
            tts_speak(VOICE_TIPS[TARGET_LANG]["exit"], TARGET_LANG)
            break

        # 窗口手动关闭
        if cv2.getWindowProperty(zh2gbk(win_title), cv2.WND_PROP_VISIBLE) < 1:
            GLOBAL_STOP_FLAG.set()
            ai_running_event.clear()
            listen_running.clear()
            with GLOBAL_TTS_LOCK:
                tts_engine.stop()
            tts_speak(VOICE_TIPS[TARGET_LANG]["exit"], TARGET_LANG)
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    with GLOBAL_TTS_LOCK:
        tts_engine.stop()
    logger.info("程序已完全退出")

if __name__ == "__main__":
    main()
