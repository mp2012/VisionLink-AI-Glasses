"""
VisionLink 多模态智能辅助眼镜 (RTX 4070 笔记本演示终极版)
修复项：
1. 麦克风并发抢占崩溃 0xC0000005
2. 空格键连击触发大量线程
3. 语音监听资源不释放问题
字体：系统黑体 simhei.ttf
功能：按键提示音 + 新语音自动终止旧语音 + 分离UI防视觉幻觉 + 自动保存干净画面
快捷键：
  1-5    切换功能
  空格    执行当前功能
  L       切换中英双语
  M       开关自动模式
  S       停止语音
  ESC     退出程序
"""
import os
import sys
import cv2
import ollama
import base64
import threading
import time
import logging
import subprocess
import numpy as np
import winsound
from PIL import Image, ImageDraw, ImageFont

# ====================== 编码 & 日志配置 ======================
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
logging.getLogger("comtypes").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} | {levelname} | {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def zh2gbk(s):
    return s.encode("gbk", errors="replace").decode("gbk")

# ====================== 创建图片保存目录 ======================
SAVE_DIR = "saved_captures"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    logger.info(f"创建图片保存文件夹: {SAVE_DIR}")

# ====================== 按键音效 ======================
def play_key_sound():
    """播放按键提示音"""
    winsound.Beep(800, 80)

# ====================== 语音全局控制 ======================
tts_process = None

def stop_all_tts():
    """停止所有正在播放的语音"""
    global tts_process
    if tts_process is not None:
        try:
            tts_process.terminate()
            tts_process.wait()
            logger.info("已终止上一条语音")
        except Exception as e:
            logger.warning(f"终止语音进程异常: {e}")
        tts_process = None

# ====================== 字体配置【系统黑体】 ======================
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
try:
    font_20 = ImageFont.truetype(FONT_PATH, 20)
    font_22 = ImageFont.truetype(FONT_PATH, 22)
    font_24 = ImageFont.truetype(FONT_PATH, 24)
    logger.info("字体加载成功")
except Exception as e:
    logger.error(f"字体加载失败: {e}")
    logger.info("尝试备用字体 simsun.ttc")
    try:
        FONT_PATH = "C:/Windows/Fonts/simsun.ttc"
        font_20 = ImageFont.truetype(FONT_PATH, 20)
        font_22 = ImageFont.truetype(FONT_PATH, 22)
        font_24 = ImageFont.truetype(FONT_PATH, 24)
        logger.info("备用宋体加载成功")
    except:
        logger.error("所有中文字体加载失败，程序退出")
        sys.exit(1)

# ====================== 全局参数 & 新增防崩溃变量 ======================
MODEL_NAME = "gemma4:e2b"
USB_CAM_ID = 1
FRAME_WIDTH = 800
FRAME_HEIGHT = 600
AI_IMAGE_SIZE = 448

TIMEOUT_ASR = 8
TIMEOUT_INFER = 12
AGENT_SCAN_INTERVAL = 5.0
BROADCAST_COOLDOWN = 15
LONG_TIME_LIMIT = 40
MAX_CONTEXT_ROUND = 3

# 状态枚举
STATE_IDLE = 0
STATE_CAPTURE = 1
STATE_LISTEN = 2
STATE_INFER = 3
STATE_TTS = 4
current_state = STATE_IDLE

current_mode = 1
voice_lang = "zh"  # zh 中文 / en 英文

agent_auto_enable = False
last_agent_run_time = 0
last_auto_broadcast = {1:0, 2:0, 3:0, 4:0, 5:0}
last_scene_task = []
same_scene_count = 0
context_history = []

# ========== 防崩溃核心新增变量 ==========
mic_listening = False        # 麦克风互斥锁：禁止并发监听
last_space_press = 0         # 空格计时
SPACE_DEBOUNCE = 0.8         # 空格防抖间隔 800ms

# ====================== 双语文本定义 ======================
MODE_NAME_LIST = {
    "zh": ["障碍物检测", "文字识别", "人脸检测", "场景介绍", "语音交互"],
    "en": ["Obstacle Detect", "Text Read", "Face Recognize", "Scene Intro", "Chat & Translate"]
}
STATE_NAME_LIST = {
    "zh": ["空闲", "采集", "聆听", "推理", "播报"],
    "en": ["Idle", "Capture", "Listening", "Inferring", "Speaking"]
}
GUIDE_TEXT = {
    "zh": {
        "title": "=== 操作指引 ===",
        "key1": "1-5  : 切换功能",
        "key2": "空格 : 立即执行",
        "key3": "L    : 切换语种",
        "key4": "M    : 开关自动模式",
        "key5": "S    : 停止语音",
        "key6": "ESC  : 退出程序",
        "pipe": "流程: 图像 -> AI -> 语音"
    },
    "en": {
        "title": "=== Operation Guide ===",
        "key1": "1-5  : Switch Mode",
        "key2": "Space: Execute Now",
        "key3": "L    : Toggle Language",
        "key4": "M    : Toggle Auto",
        "key5": "S    : Stop Voice",
        "key6": "ESC  : Exit",
        "pipe": "Pipeline: Image -> AI -> Voice"
    }
}
TIP_VOICE = {
    "zh": {
        "start": "系统启动完成",
        "auto_on": "自动助手已开启",
        "auto_off": "自动助手已关闭",
        "lang_switch_en": "语音已切换为英文",
        "lang_switch_zh": "语音已切换为中文",
        "stop_voice": "语音已停止",
        "no_voice": "当前无语音",
        "no_change": "画面无变化，减少提醒",
        "cmd_ok": "收到指令，开始执行",
        "mic_fail": "无法识别语音，请重试",
        "exit": "程序即将退出"
    },
    "en": {
        "start": "System started",
        "auto_on": "Auto agent enabled",
        "auto_off": "Auto agent disabled",
        "lang_switch_en": "Voice switched to English",
        "lang_switch_zh": "Voice switched to Chinese",
        "stop_voice": "Voice stopped",
        "no_voice": "No voice playing",
        "no_change": "Same scene, less reminder",
        "cmd_ok": "Command received, executing",
        "mic_fail": "Failed to recognize voice, please try again",
        "exit": "Program exiting"
    }
}
PROMPT_LIB = {
    "zh": [
        "你是智能辅助眼镜的障碍物提醒助手，用简短一句话中文提醒道路风险。",
        "你是一个盲人文字读取器。请直接读出画面中物品上的核心中文印刷文字，严禁输出任何问候语、分析过程或多余废话，直接输出正文！",
        "检测画面中的人脸并给出简短提示。",
        "用简短中文介绍眼前风景场景。",
        "结合图片回答问题，支持中英互译，回答简洁。"
    ],
    "en": [
        "You are an obstacle reminder for smart glasses, warn road hazards briefly in English.",
        "Read all printed text in the picture briefly in English. Do not output any notes, just text.",
        "Detect faces and give a short prompt in English.",
        "Briefly introduce the scene in English.",
        "Answer questions with image, support translation, keep answer short in English."
    ]
}
AGENT_PROMPT = """
You are a multi-modal task scheduler for smart glasses.
Only output number, no extra words:
1=Obstacle detection
2=Read text
3=Recognize face
4=Introduce scene
5=Chat & translate
Separate multiple numbers with comma.
"""
TASK_PLAN_PROMPT = """
Split user command into tasks, only output numbers:
1=Obstacle 2=Read text 3=Face 4=Scene 5=Chat
Split with comma, no extra words.
User command: {user_cmd}
"""

# ====================== 工具函数 ======================
def save_clean_frame(frame, mode_num):
    """自动保存当前识别的干净图片"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"mode_{mode_num}_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)
    logger.info(f"📷 干净图片已自动保存至: {filepath}")

def image_to_base64(frame):
    logger.info("开始转换图像为Base64编码")
    resize_img = cv2.resize(frame, (AI_IMAGE_SIZE, AI_IMAGE_SIZE))
    success, buf = cv2.imencode(".jpg", resize_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not success:
        logger.error("图像编码失败")
        return None
    b64_data = base64.b64encode(buf).decode("utf-8")
    logger.info("图像Base64转换完成")
    return b64_data

def gemma_infer(prompt: str, img_base64=None):
    global current_state
    if current_state != STATE_IDLE:
        logger.warning("系统忙，拒绝本次推理请求")
        return ""
    try:
        current_state = STATE_INFER
        logger.info("开始调用大模型推理")
        messages = [{"role": "user", "content": prompt}]
        if img_base64:
            messages[0]["images"] = [img_base64]
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"timeout": TIMEOUT_INFER * 1000}
        )
        res_text = resp["message"]["content"].strip()
        logger.info(f"大模型推理完成，返回内容：{res_text}")
        return res_text
    except Exception as e:
        logger.error(f"大模型推理异常：{str(e)}")
        return ""
    finally:
        current_state = STATE_IDLE
        logger.info("推理状态已重置为空闲")

def stop_tts():
    """手动停止语音（S键调用）"""
    stop_all_tts()
    tip = TIP_VOICE[voice_lang]["stop_voice"]
    cmd = (
        f'powershell "Add-Type -AssemblyName System.Speech;'
        f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
        f'$synth.Rate = -2;'
        f'$synth.Speak(\'{tip}\')"'
    )
    global tts_process
    tts_process = subprocess.Popen(cmd, shell=True)
    logger.info(f"播放停止提示语音：{tip}")

def tts_speak(text: str):
    """播放语音：先停旧语音，再播新语音"""
    global tts_process
    if not text.strip():
        logger.info("语音内容为空，跳过播放")
        return
    # 核心逻辑：新语音到来，先终止所有正在播放的语音
    stop_all_tts()

    logger.info(f"准备播放语音，原始内容：{text}")
    clean_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    clean_text = " ".join(clean_text.split())
    safe_text = clean_text.replace("'", "''")

    cmd = (
        f'powershell "Add-Type -AssemblyName System.Speech;'
        f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
        f'$synth.Rate = -2;'
        f'$synth.Speak(\'{safe_text}\')"'
    )
    try:
        tts_process = subprocess.Popen(cmd, shell=True)
        logger.info(f"语音已开始播放：{clean_text}")
    except Exception as e:
        logger.error(f"语音播放异常：{str(e)}")
        tts_process = None

def switch_voice_language():
    global voice_lang
    logger.info("检测到L键，切换语种")
    if voice_lang == "zh":
        voice_lang = "en"
        tip = TIP_VOICE["en"]["lang_switch_en"]
    else:
        voice_lang = "zh"
        tip = TIP_VOICE["zh"]["lang_switch_zh"]
    logger.info(f"当前语种切换为：{voice_lang}")
    tts_speak(tip)

# ====================== 功能函数（修复麦克风并发） ======================
def func_obstacle(frame):
    logger.info("执行功能1：障碍物检测提醒")
    save_clean_frame(frame, 1)
    img_b64 = image_to_base64(frame)
    res = gemma_infer(PROMPT_LIB[voice_lang][0], img_b64)
    if res:
        tts_speak(res)

def func_ocr(frame):
    logger.info("执行功能2：文字识别朗读")
    save_clean_frame(frame, 2)
    img_b64 = image_to_base64(frame)
    res = gemma_infer(PROMPT_LIB[voice_lang][1], img_b64)
    if res:
        tts_speak(res)

def func_face(frame):
    logger.info("执行功能3：人脸检测")
    save_clean_frame(frame, 3)
    img_b64 = image_to_base64(frame)
    res = gemma_infer(PROMPT_LIB[voice_lang][2], img_b64)
    if res:
        tts_speak(res)

def func_scene(frame):
    logger.info("执行功能4：场景介绍")
    save_clean_frame(frame, 4)
    img_b64 = image_to_base64(frame)
    res = gemma_infer(PROMPT_LIB[voice_lang][3], img_b64)
    if res:
        tts_speak(res)

def func_chat(frame):
    global context_history, current_state, mic_listening
    # 麦克风互斥：正在监听则直接返回，拒绝并发
    if mic_listening:
        logger.warning("麦克风正在使用，忽略本次语音交互触发")
        return

    logger.info("执行功能5：语音交互")
    save_clean_frame(frame, 5)
    current_state = STATE_LISTEN
    mic_listening = True  # 加锁
    logger.info("开始监听麦克风语音")

    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        # 单次麦克风实例，不重复创建
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, 0.3)
            audio = r.listen(source, timeout=TIMEOUT_ASR)

        current_state = STATE_IDLE
        user_text = r.recognize_google(audio, language="zh-CN")
        logger.info(f"麦克风识别内容：{user_text}")

        task_raw = gemma_infer(TASK_PLAN_PROMPT.format(user_cmd=user_text))
        task_list = []
        try:
            num_strs = task_raw.strip().split(",")
            for s in num_strs:
                n = int(s.strip())
                if 1 <= n <= 5:
                    task_list.append(n)
        except:
            task_list = []

        if task_list:
            logger.info(f"解析出自动任务列表：{task_list}")
            tts_speak(TIP_VOICE[voice_lang]["cmd_ok"])
            agent_auto_execute(task_list, frame)
            return

        img_b64 = image_to_base64(frame)
        history_str = ""
        for item in context_history:
            history_str += f"User: {item['q']}\nAssistant: {item['a']}\n"
        full_prompt = f"{history_str}\n{PROMPT_LIB[voice_lang][4]}\nUser: {user_text}"
        ans = gemma_infer(full_prompt, img_b64)
        if ans:
            tts_speak(ans)
            context_history.append({"q": user_text, "a": ans})
            if len(context_history) > MAX_CONTEXT_ROUND:
                context_history.pop(0)

    except Exception as e:
        current_state = STATE_IDLE
        logger.warning(f"语音识别异常：{str(e)}")
        tts_speak(TIP_VOICE[voice_lang]["mic_fail"])
    finally:
        # 无论正常/异常，强制解锁、复位状态
        mic_listening = False
        current_state = STATE_IDLE
        logger.info("麦克风监听已结束，锁已释放")

def agent_scene_detect(frame) -> list:
    logger.info("自动模式：开始场景任务分配推理")
    img_b64 = image_to_base64(frame)
    res = gemma_infer(AGENT_PROMPT, img_b64)
    task_list = []
    try:
        num_strs = res.strip().split(",")
        for s in num_strs:
            n = int(s.strip())
            if 1 <= n <= 5:
                task_list.append(n)
    except:
        task_list = []
    logger.info(f"自动分配任务：{task_list}")
    return task_list

def agent_auto_execute(task_nums: list, frame):
    global last_auto_broadcast, last_scene_task, same_scene_count
    now = time.time()
    if task_nums == last_scene_task and len(task_nums) > 0:
        same_scene_count += 1
        logger.info(f"画面无变化，累计静默次数：{same_scene_count}")
        if same_scene_count >= 3:
            for num in task_nums:
                last_auto_broadcast[num] = now + LONG_TIME_LIMIT
            tts_speak(TIP_VOICE[voice_lang]["no_change"])
            return
    else:
        same_scene_count = 0
        last_scene_task = task_nums.copy()

    func_map = {1: func_obstacle, 2: func_ocr, 3: func_face, 4: func_scene, 5: func_chat}
    run_tasks = task_nums[:2]
    logger.info(f"自动模式执行任务（限2个）：{run_tasks}")
    for num in run_tasks:
        if now < last_auto_broadcast[num] or current_state != STATE_IDLE:
            logger.info(f"任务{num}未到执行时间/系统忙碌，跳过")
            continue
        f = func_map.get(num)
        if f:
            threading.Thread(target=f, args=(frame.copy(),), daemon=True).start()
            last_auto_broadcast[num] = now + BROADCAST_COOLDOWN
            time.sleep(1.0)

# ====================== UI 绘制 ======================
def draw_ui_tips(frame):
    global current_state, current_mode, agent_auto_enable, voice_lang
    h, w = frame.shape[:2]
    panel_w = 320

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
    alpha = 0.65
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    y = 30

    color_map = {
        STATE_IDLE: (0, 220, 0),
        STATE_CAPTURE: (220, 150, 0),
        STATE_LISTEN: (255, 100, 0),
        STATE_INFER: (160, 0, 220),
        STATE_TTS: (0, 100, 255)
    }
    draw.rectangle([10, y-15, 30, y+5], fill=color_map[current_state])
    status_txt = STATE_NAME_LIST[voice_lang][current_state]
    if voice_lang == "zh":
        draw.text((45, y-12), f"状态：{status_txt}", font=font_24, fill=(255,255,255), align="left")
    else:
        draw.text((45, y-12), f"Status: {status_txt}", font=font_24, fill=(255,255,255), align="left")
    y += 45

    mode_txt = MODE_NAME_LIST[voice_lang][current_mode - 1]
    if voice_lang == "zh":
        draw.text((15, y-12), f"模式 {current_mode}：{mode_txt}", font=font_22, fill=(0,255,255), align="left")
    else:
        draw.text((15, y-12), f"Mode {current_mode}: {mode_txt}", font=font_22, fill=(0,255,255), align="left")
    y += 40

    if agent_auto_enable:
        auto_fill = (0,255,0)
        auto_txt = "自动模式：已开启(按M关闭)" if voice_lang == "zh" else "Auto Agent: ON (M=OFF)"
    else:
        auto_fill = (120,120,120)
        auto_txt = "自动模式：已关闭(按M开启)" if voice_lang == "zh" else "Auto Agent: OFF (M=ON)"
    draw.text((15, y-12), auto_txt, font=font_22, fill=auto_fill, align="left")
    y += 40

    lang_txt = "当前语种：中文" if voice_lang == "zh" else "Language: English"
    draw.text((15, y-12), lang_txt, font=font_22, fill=(255, 200, 0), align="left")
    y += 45

    g = GUIDE_TEXT[voice_lang]
    draw.text((15, y-12), g["title"], font=font_22, fill=(255,255,255), align="left")
    y += 35
    draw.text((15, y-10), g["key1"], font=font_20, fill=(255,255,255), align="left")
    y += 30
    draw.text((15, y-10), g["key2"], font=font_20, fill=(255,255,255), align="left")
    y += 30
    draw.text((15, y-10), g["key3"], font=font_20, fill=(255, 200, 0), align="left")
    y += 30
    draw.text((15, y-10), g["key4"], font=font_20, fill=(0,255,0), align="left")
    y += 30
    draw.text((15, y-10), g["key5"], font=font_20, fill=(255,0,0), align="left")
    y += 30
    draw.text((15, y-10), g["key6"], font=font_20, fill=(220,220,220), align="left")
    y += 50

    draw.text((15, y-10), g["pipe"], font=font_20, fill=(200,200,200), align="left")
    y = h - 25
    draw.text((15, y-10), "VisionLink | Edge AI", font=font_20, fill=(150,150,150), align="left")

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame

# ====================== 主程序入口（增加空格防抖） ======================
def main():
    global current_mode, agent_auto_enable, last_agent_run_time, last_space_press
    logger.info("========== 程序启动 ==========")
    tts_speak(TIP_VOICE[voice_lang]["start"])

    logger.info("初始化摄像头")
    cap = cv2.VideoCapture(USB_CAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("CAP_DSHOW 打开失败，使用默认摄像头模式")
        cap = cv2.VideoCapture(USB_CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    logger.info(f"摄像头初始化完成，分辨率：{FRAME_WIDTH}×{FRAME_HEIGHT}")

    win_name = zh2gbk("VisionLink Smart Glass")
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, FRAME_WIDTH, FRAME_HEIGHT)
    logger.info("窗口创建完成，非全屏模式")

    func_map = {1: func_obstacle, 2: func_ocr, 3: func_face, 4: func_scene, 5: func_chat}
    logger.info("功能映射表加载完毕，进入主循环")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("读取摄像头画面失败")
            time.sleep(0.1)
            continue

        # 拷贝纯净无UI画面用于识别、推理、保存
        raw_frame = frame.copy()
        # 绘制UI仅用于屏幕预览
        ui_frame = draw_ui_tips(frame)

        now_t = time.time()
        if agent_auto_enable and now_t - last_agent_run_time >= AGENT_SCAN_INTERVAL and current_state == STATE_IDLE:
            logger.info("自动模式触发，开始新一轮场景检测")
            last_agent_run_time = now_t
            tasks = agent_scene_detect(raw_frame)
            if tasks:
                threading.Thread(target=agent_auto_execute, args=(tasks, raw_frame.copy()), daemon=True).start()

        cv2.imshow(win_name, ui_frame)
        key = cv2.waitKey(20) & 0xFF

        # S 键 停止语音
        if key == ord('s') or key == ord('S'):
            play_key_sound()
            stop_tts()
            continue

        # M 键 切换自动模式
        if key == ord('m') or key == ord('M'):
            play_key_sound()
            logger.info("检测到M键，切换自动模式状态")
            agent_auto_enable = not agent_auto_enable
            if agent_auto_enable:
                logger.info("自动助手已开启")
                tts_speak(TIP_VOICE[voice_lang]["auto_on"])
            else:
                logger.info("自动助手已关闭")
                tts_speak(TIP_VOICE[voice_lang]["auto_off"])
            continue

        # L 键 切换语种
        if key == ord('l') or key == ord('L'):
            play_key_sound()
            switch_voice_language()
            continue

        # 1~5 切换功能
        if 49 <= key <= 53:
            play_key_sound()
            current_mode = key - 48
            logger.info(f"切换至功能模式 {current_mode}")
            mode_txt = MODE_NAME_LIST[voice_lang][current_mode - 1]
            if voice_lang == "en":
                tts_speak(f"Switched to {mode_txt} mode")
            else:
                tts_speak(f"已切换至 {mode_txt} 模式")
            continue

        # 空格键 + 防抖处理（核心防连击炸线程）
        if key == 32:
            # 防抖判断
            if now_t - last_space_press < SPACE_DEBOUNCE:
                continue
            last_space_press = now_t

            play_key_sound()
            logger.info(f"空格键触发，手动执行模式 {current_mode}")
            f = func_map.get(current_mode)
            threading.Thread(target=f, args=(raw_frame.copy(),), daemon=True).start()
            continue

        # ESC 退出
        if key == 27:
            play_key_sound()
            logger.info("检测到ESC键，准备退出程序")
            stop_all_tts()
            tts_speak(TIP_VOICE[voice_lang]["exit"])
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("摄像头与窗口资源已释放，程序结束")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序全局异常：{str(e)}")
