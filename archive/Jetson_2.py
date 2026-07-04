"""
VisionLink Jetson 纯手动开发版
运行环境：Jetson Orin Nano + Ubuntu + PyCharm + USB摄像头/键鼠/耳麦
核心改动：删除全部自动障碍扫描，仅空格键手动识别，杜绝后台持续算力占用卡顿
功能清单：
1. 纯净无文字摄像头预览窗口，画面无任何叠加UI，信息仅控制台日志输出
2. 空格键：抓拍原图保存 + 单次AI多模态推理（仅手动触发，无自动循环）
3. S键：强制终止全部正在播放语音，新语音自动覆盖旧语音无杂音
4. 自动创建三类存储目录：
   - snapshots/ 抓拍原图（时间戳jpg）
   - infer_logs/ 大模型推理文本，每条独立txt
   - audio_cache/ 语音播报wav音频文件
5. 推理12秒超时兜底，解决gemma4:e2b慢速推理锁死
6. 全流程详细分级开发日志，PyCharm控制台查看所有状态、报错、结果
7. 可用快捷键：1-5切换识别模式、空格手动识别、S停止语音、ESC安全退出
"""
import os
import sys

# 屏蔽Qt字体stderr警告
devnull = open(os.devnull, 'w')
sys.stderr = devnull

# Qt环境变量抑制字体警告
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu/"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

import cv2
import ollama
import base64
import threading
import time
import logging
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ====================== 全局目录初始化 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
INFER_LOG_DIR = os.path.join(BASE_DIR, "infer_logs")
AUDIO_CACHE_DIR = os.path.join(BASE_DIR, "audio_cache")

# 自动创建存储文件夹
for dir_path in [SNAPSHOT_DIR, INFER_LOG_DIR, AUDIO_CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ====================== 日志配置（唯一信息查看渠道） ======================
sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="{asctime} | {levelname:8} | {funcName:15} | {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 全局语音进程句柄
tts_process = None

# ====================== Jetson 中文字体加载（仅用于txt文件写入，画面无绘制） ======================
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
font_20 = font_22 = font_24 = None
try:
    font_20 = ImageFont.truetype(FONT_PATH, 20)
    font_22 = ImageFont.truetype(FONT_PATH, 22)
    font_24 = ImageFont.truetype(FONT_PATH, 24)
    logger.info("✅ 中文字体加载成功 wqy-microhei（仅文本存储使用，画面无UI）")
except Exception as e:
    logger.error(f"❌ 字体缺失，请执行 sudo apt install fonts-wqy-microhei | {e}")
    sys.exit(1)

# ====================== 全局配置参数（已删除所有自动扫描相关常量） ======================
MODEL_NAME = "gemma4:e2b"
FRAME_WIDTH = 800
FRAME_HEIGHT = 600
AI_IMAGE_SIZE = 384
JPEG_QUALITY = 70
TIMEOUT_INFER = 12

# 推理状态锁，防止并发卡死
STATE_IDLE = 0
STATE_INFER = 3
current_state = STATE_IDLE

# 当前识别模式
current_mode = 1

# 模式名称与对应提示词
MODE_NAMES = [
    "1 障碍物检测",
    "2 文字识别朗读",
    "3 人脸检测提示",
    "4 场景简短描述",
    "5 图文问答交互"
]
PROMPT_LIB = [
    "你是盲人眼镜避障助手，观察画面中心，识别近处障碍物并估算距离，25字内纯中文。",
    "识别图片中全部印刷文字，精简读出，不要多余描述。",
    "检测画面里所有人脸，简单描述人脸位置与数量。",
    "简短描述眼前场景环境，控制在50字以内，温柔口语。",
    "结合图片回答问题，语言简洁易懂。"
]

# ====================== 工具函数：按键短促蜂鸣提示 ======================
def key_beep():
    os.system("speaker-test -t sine -f 900 -l 0.06 >/dev/null 2>&1")

# ====================== TTS语音控制：新语音自动杀死旧语音，保存wav ======================
def stop_all_tts():
    global tts_process
    if tts_process is not None:
        try:
            tts_process.terminate()
            tts_process.wait(timeout=0.8)
            logger.info("🔇 已终止上一段语音播放进程")
        except Exception as e:
            logger.warning(f"终止语音进程异常: {str(e)}")
        tts_process = None

def tts_speak(text: str):
    global tts_process
    text = text.strip()
    if not text:
        logger.warning("语音文本为空，跳过播报")
        return
    stop_all_tts()

    timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
    wav_save_path = os.path.join(AUDIO_CACHE_DIR, f"voice_{timestamp}.wav")
    clean_text = text.replace("\n", " ").replace("'", " ")

    espeak_cmd = (
        f"espeak -v zh+f2 -s 160 --wav={wav_save_path} '{clean_text}' >/dev/null 2>&1;"
        f"aplay {wav_save_path} >/dev/null 2>&1"
    )
    try:
        tts_process = subprocess.Popen(espeak_cmd, shell=True)
        logger.info(f"🔊 语音播报 | 文本：{text} | 音频保存路径：{wav_save_path}")
    except Exception as e:
        logger.error(f"语音生成/播放失败: {str(e)}")

def stop_voice_hint():
    stop_all_tts()
    tts_speak("语音已停止")

# ====================== 图像存储工具 ======================
def save_snapshot(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
    img_path = os.path.join(SNAPSHOT_DIR, f"snap_{timestamp}.jpg")
    cv2.imwrite(img_path, frame)
    logger.info(f"📸 抓拍图片保存完成：{img_path}")
    return img_path

def image_to_base64(frame):
    resize_img = cv2.resize(frame, (AI_IMAGE_SIZE, AI_IMAGE_SIZE))
    ok, buf = cv2.imencode(".jpg", resize_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        logger.error("图像Base64编码失败")
        return None
    return base64.b64encode(buf).decode("utf-8")

# ====================== 推理核心：仅空格键手动调用 ======================
def run_infer_task(frame, mode_idx):
    global current_state
    if current_state != STATE_IDLE:
        logger.warning(f"当前推理未完成，忽略本次模式{mode_idx}手动识别请求")
        return
    current_state = STATE_INFER
    snapshot_path = save_snapshot(frame)
    logger.info(f"🧠 手动触发模式{mode_idx}推理，抓拍原图：{snapshot_path}")

    def infer_work():
        result_text = ""
        try:
            img_b64 = image_to_base64(frame)
            prompt = PROMPT_LIB[mode_idx - 1]
            logger.info(f"发送多模态请求，Prompt：{prompt}")
            resp = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
                options={"timeout": TIMEOUT_INFER * 1000}
            )
            result_text = resp["message"]["content"].strip()
            logger.info(f"✅ 大模型推理完成，返回结果：{result_text}")

            # 保存推理文本到独立txt
            timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
            txt_path = os.path.join(INFER_LOG_DIR, f"infer_{timestamp}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"推理时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"运行模式：{MODE_NAMES[mode_idx-1]}\n")
                f.write(f"抓拍原图路径：{snapshot_path}\n")
                f.write(f"模型：{MODEL_NAME}\n")
                f.write("="*40 + "\n")
                f.write(result_text)
            logger.info(f"📄 推理文本结果已保存至：{txt_path}")

            # 语音播报并自动保存wav
            tts_speak(result_text)

        except Exception as e:
            err_msg = f"AI推理异常：{str(e)}"
            logger.error(err_msg)
            tts_speak("AI识别超时失败")
        finally:
            global current_state
            current_state = STATE_IDLE
            logger.info("单次手动推理流程结束，系统恢复空闲")

    worker = threading.Thread(target=infer_work, daemon=True)
    worker.start()
    # 超时强制释放锁，防止慢速模型永久占用
    worker.join(timeout=TIMEOUT_INFER)
    if current_state != STATE_IDLE:
        current_state = STATE_IDLE
        logger.warning(f"⚠️ 推理线程超时{TIMEOUT_INFER}秒，强制释放空闲锁")

# ====================== 画面渲染：纯原图无任何UI文字 ======================
def draw_info_panel(frame):
    return frame

# ====================== 主循环入口 ======================
def main():
    global current_mode
    logger.info("=" * 60)
    logger.info("VisionLink Jetson 纯手动开发版 程序启动")
    logger.info(f"模型：{MODEL_NAME} | 预览分辨率 {FRAME_WIDTH}×{FRAME_HEIGHT}")
    logger.info(f"图片保存：{SNAPSHOT_DIR}")
    logger.info(f"推理文本保存：{INFER_LOG_DIR}")
    logger.info(f"语音音频保存：{AUDIO_CACHE_DIR}")
    logger.info("运行模式：仅空格键手动AI识别，无自动后台扫描，低负载不卡顿")
    logger.info("=" * 60)
    tts_speak("系统启动完成，按空格键进行图像识别")

    # 自动遍历0~5寻找可用摄像头，适配/dev/video1
    cap = None
    found_cam_id = -1
    for cam_idx in range(6):
        temp_cap = cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            found_cam_id = cam_idx
            temp_cap.release()
            break
    if found_cam_id == -1:
        logger.error("遍历0~5所有摄像头均无法打开，程序退出")
        return
    cap = cv2.VideoCapture(found_cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    logger.info(f"自动检测到可用摄像头ID：{found_cam_id}")

    # 创建纯净预览窗口
    win_title = "VisionLink Pure Preview - Manual Only"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, FRAME_WIDTH, FRAME_HEIGHT)
    logger.info("纯净预览窗口创建成功，画面无任何文字UI")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("摄像头读取帧失败，短暂重试")
                time.sleep(0.05)
                continue

            display_frame = draw_info_panel(frame.copy())
            cv2.imshow(win_title, display_frame)

            # 键盘捕获
            key = cv2.waitKey(20) & 0xFF

            # ESC 安全退出
            if key == 27:
                key_beep()
                logger.info("检测ESC按键，执行退出流程")
                stop_all_tts()
                tts_speak("程序即将退出")
                break

            # S / s 停止全部语音
            if key == ord("s") or key == ord("S"):
                key_beep()
                logger.info("按下S键，终止所有语音播报")
                stop_voice_hint()
                continue

            # 1~5 切换识别模式
            if 49 <= key <= 53:
                key_beep()
                current_mode = int(chr(key))
                tip_text = f"已切换至{MODE_NAMES[current_mode - 1]}"
                logger.info(tip_text)
                tts_speak(tip_text)
                continue

            # 空格键：手动抓拍+单次AI推理（唯一触发识别入口）
            if key == 32:
                key_beep()
                logger.info("空格键按下，手动抓拍并执行AI推理")
                threading.Thread(target=run_infer_task, args=(frame.copy(), current_mode), daemon=True).start()
                continue

    except KeyboardInterrupt:
        logger.info("捕获Ctrl+C强制中断，开始释放资源")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_all_tts()
        logger.info("摄像头、窗口、语音进程全部释放，程序正常结束")

if __name__ == "__main__":
    main()
