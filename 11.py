"""
VisionLink-AI-Glasses
修复：pyttsx3 单次发声后后续静默、语言切换语音包不匹配
特性：全操作语音提示、窗口中文不乱码、完整日志、线程安全
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

# ==================== 全局配置 ====================
TARGET_MODEL = 'gemma4:e2b'
USB_CAMERA_ID = 1
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
AI_IMAGE_SIZE = 448
FRAME_CROP_RATIO = 0.25

MODE_TIPS = {
    "zh": ["避障模式", "文字阅读模式", "场景描述模式"],
    "en": ["Obstacle mode", "Text reading mode", "Scene description mode"]
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
        "exit": "程序退出"
    },
    "en": {
        "system_start": "System started",
        "busy": "Processing, please wait",
        "capture": "Start recognition",
        "recognize_fail": "Recognition failed",
        "exit": "Program exit"
    }
}
TARGET_LANG = 'zh'
# ==========================================================

current_mode = 1
ai_running_event = threading.Event()
GLOBAL_TTS_LOCK = threading.Lock()

# ==================== 全新语音方案：每次播报临时创建引擎（解决只响一次） ====================
def tts_speak(text, lang="zh"):
    """独立语音函数，每次新建引擎，彻底解决后续无声问题"""
    if not text:
        return
    def _run():
        with GLOBAL_TTS_LOCK:
            try:
                logger.info(f"开始语音播报: {text}")
                engine = pyttsx3.init()
                engine.setProperty('rate', 180)
                engine.setProperty('volume', 1.0)
                voices = engine.getProperty('voices')

                # 精准匹配语音包
                for v in voices:
                    name = v.name.lower()
                    vid = v.id.lower()
                    if lang == "zh":
                        if "chinese" in name or "zh" in vid:
                            engine.setProperty('voice', v.id)
                            break
                    else:
                        if "english" in name or "en" in vid:
                            engine.setProperty('voice', v.id)
                            break

                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
                logger.info("语音播报完成")
            except Exception as e:
                logger.error(f"语音异常: {str(e)}")
    threading.Thread(target=_run, daemon=True).start()

# ==================== 字体加载 ====================
def load_system_font():
    logger.info("加载系统字体")
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc"
    ]
    font_lg = None
    font_sm = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font_lg = ImageFont.truetype(path, 24)
                font_sm = ImageFont.truetype(path, 18)
                logger.info(f"成功加载字体: {path}")
                break
            except:
                logger.warning(f"字体 {path} 加载失败")
    if not font_lg:
        logger.warning("未找到中文字体，使用默认字体")
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

def shutter():
    try:
        logger.info("播放快门音效")
        threading.Thread(
            target=lambda: ctypes.windll.winmm.PlaySoundW("SystemAsterisk", 0, 1 | 0x400000),
            daemon=True
        ).start()
    except Exception:
        logger.warning("快门音效播放失败")

def analyze_frame(frame, mode, lang):
    global ai_running_event
    logger.info(f"进入AI分析线程，模式: {mode}, 语言: {lang}")
    try:
        h, w = frame.shape[:2]
        logger.info(f"画面尺寸: 宽={w}, 高={h}")
        if h == 0 or w == 0:
            logger.error("画面为空")
            tts_speak(VOICE_TIPS[lang]["recognize_fail"], lang)
            return

        scale = AI_IMAGE_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        logger.info(f"缩放后尺寸: {new_w} x {new_h}")

        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]
        ret, buf = cv2.imencode(".jpg", resized, encode_param)
        if not ret:
            logger.error("图像编码失败")
            tts_speak(VOICE_TIPS[lang]["recognize_fail"], lang)
            return
        b64_data = base64.b64encode(buf).decode()
        logger.info("图像Base64编码完成")

        if mode == 1:
            prompt = "前方障碍物是什么？距离多远？简短回答。"
        elif mode == 2:
            prompt = "提取图片所有文字，只输出文字。"
        else:
            prompt = "描述眼前场景，简洁。"
        logger.info(f"使用提示词: {prompt}")

        resp = ollama.chat(
            model=TARGET_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [b64_data]}],
            options={"temperature": 0.1}
        )
        logger.info("Ollama 推理完成")

        if resp and "message" in resp:
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
        logger.info("AI线程结束")

def main():
    global current_mode, ai_running_event, TARGET_LANG
    logger.info("===== 程序启动 =====")
    logger.info(f"摄像头ID={USB_CAMERA_ID}, 分辨率={PREVIEW_WIDTH}x{PREVIEW_HEIGHT}, 模型={TARGET_MODEL}")

    logger.info(f"尝试打开摄像头 {USB_CAMERA_ID}")
    cap = cv2.VideoCapture(USB_CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("摄像头1打开失败，切换为0")
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"摄像头就绪，实际分辨率: {actual_w} x {actual_h}")

    win_title = "VisionLink AI眼镜"
    cv2.namedWindow(zh2gbk(win_title), cv2.WINDOW_NORMAL)
    # 启动语音
    tts_speak(VOICE_TIPS[TARGET_LANG]["system_start"], TARGET_LANG)

    logger.info("进入画面循环")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("读取帧失败")
            continue

        h, w = frame.shape[:2]
        crop_pad = int(min(w, h) * FRAME_CROP_RATIO)
        bx1, by1 = crop_pad, crop_pad
        bx2, by2 = w - crop_pad, h - crop_pad

        if current_mode == 2:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

        cv2.rectangle(frame, (20, 20), (650, 80), (0, 0, 0), -1)
        status_text = f"模式：{MODE_TIPS[TARGET_LANG][current_mode-1]} | 语言：{TARGET_LANG.upper()}"
        frame = draw_text(frame, status_text, (35, 35), font_lg)

        if ai_running_event.is_set():
            cv2.rectangle(frame, (w-280, h-70), (w-20, h-20), (0, 0, 255), -1)
            frame = draw_text(frame, "识别中...", (w-250, h-58), font_sm, (255, 255, 255))

        tip_text = "空格:识别 | 1/2/3:模式 | L:语言 | ESC:退出"
        frame = draw_text(frame, tip_text, (30, h-45), font_sm, (255, 255, 255))

        cv2.imshow(zh2gbk(win_title), frame)
        key = cv2.waitKey(15) & 0xFF

        # ESC 退出
        if key == 27:
            logger.info("按下ESC，准备退出")
            tts_speak(VOICE_TIPS[TARGET_LANG]["exit"], TARGET_LANG)
            break

        # 空格识别
        elif key == ord(' '):
            logger.info("按下空格，触发识别")
            if not ai_running_event.is_set():
                ai_running_event.set()
                shutter()
                tts_speak(VOICE_TIPS[TARGET_LANG]["capture"], TARGET_LANG)
                if current_mode == 2 and (by2 > by1 and bx2 > bx1):
                    snap_frame = frame[by1:by2, bx1:bx2].copy()
                    logger.info("文字模式：截取中心区域")
                else:
                    snap_frame = frame.copy()
                    logger.info("全画面识别")
                threading.Thread(
                    target=analyze_frame,
                    args=(snap_frame, current_mode, TARGET_LANG),
                    daemon=True
                ).start()
            else:
                logger.warning("AI忙，忽略请求")
                tts_speak(VOICE_TIPS[TARGET_LANG]["busy"], TARGET_LANG)

        # 切换模式 1/2/3
        elif key in (ord('1'), ord('2'), ord('3')):
            new_mode = int(chr(key))
            if new_mode != current_mode:
                current_mode = new_mode
                mode_name = MODE_TIPS[TARGET_LANG][current_mode - 1]
                logger.info(f"切换模式: {mode_name}")
                tts_speak(mode_name, TARGET_LANG)
            else:
                logger.info("当前已是该模式")

        # 切换语言 L
        elif key in (ord('l'), ord('L')):
            logger.info("按下L，切换语言")
            TARGET_LANG = "en" if TARGET_LANG == "zh" else "zh"
            tts_speak(LANG_TIPS[TARGET_LANG], TARGET_LANG)

        # 窗口手动关闭
        if cv2.getWindowProperty(zh2gbk(win_title), cv2.WND_PROP_VISIBLE) < 1:
            logger.info("窗口被关闭")
            tts_speak(VOICE_TIPS[TARGET_LANG]["exit"], TARGET_LANG)
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("===== 程序完全退出 =====")

if __name__ == "__main__":
    main()
