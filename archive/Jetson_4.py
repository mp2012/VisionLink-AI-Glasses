import time
import cv2
import base64
import ollama
import numpy as np
import os
import glob
import threading
from queue import Queue

# ==================== 核心参数配置 ====================
MODEL_NAME = 'gemma4:e2b'
CAPTURE_INTERVAL = 4       # 雷打不动每 4 秒抓拍一次

DEBUG_DIR = "/home/seeed/AI/pycharm/VisionLink-AI-Glasses/VisionLink-AI-Glasses-master/debug_logs"
MAX_SAVE_FILES = 100

SYSTEM_PROMPT = (
    "你是一个盲人出行实时导航助手。请极其敏锐地指出图片正前方影响盲人行进的障碍物或警示牌。"
    "严格遵守以下三条铁律：\n"
    "1. 字数必须控制在 15 个字以内！\n"
    "2. 只说正前方最危险的一处情况，不要列举，不要说‘图片显示’等废话！\n"
    "3. 如果前方安全，只需回复‘安全’，绝不多吐一个字！"
)

image_queue = Queue(maxsize=1)
# ====================================================

def rotate_debug_logs():
    img_files = sorted(glob.glob(os.path.join(DEBUG_DIR, "shot_*.jpg")), key=os.path.getmtime)
    if len(img_files) > MAX_SAVE_FILES:
        excess_count = len(img_files) - MAX_SAVE_FILES
        for i in range(excess_count):
            old_img = img_files[i]
            old_txt = old_img.replace(".jpg", ".txt")
            try:
                if os.path.exists(old_img): os.remove(old_img)
                if os.path.exists(old_txt): os.remove(old_txt)
            except: pass

def ai_inference_worker():
    """ 线程 2：独立后台大脑 """
    while True:
        frame, timestamp, img_path, txt_path = image_queue.get()
        
        try:
            cv2.imwrite("live_debug.jpg", frame)
            cv2.imwrite(img_path, frame)

            _, buffer = cv2.imencode('.jpg', frame)
            b64_image = base64.b64encode(buffer).decode('utf-8')

            # 打印清爽的边界头
            print(f"\n[🧠 大脑分析中] 针对样本: shot_{timestamp}.jpg")
            print("--------------------------------------------------")
            print("🔊 语音播报 -> ", end='', flush=True)

            full_response = ""
            response_stream = ollama.generate(
                model=MODEL_NAME, prompt=SYSTEM_PROMPT, images=[b64_image],
                options={'temperature': 0.1}, stream=True
            )

            # 此时快门线程不再刷屏，这里吐字将如丝般顺滑
            for chunk in response_stream:
                word = chunk['response']
                print(word, end='', flush=True)
                full_response += word
            print("\n--------------------------------------------------")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {SYSTEM_PROMPT}\nAI Response: {full_response}\n")

            rotate_debug_logs()

        except Exception as e:
            print(f"\n❌ 后台 AI 推理链路异常: {e}")
        
        image_queue.task_done()

def main():
    print("🚀 VisionLink-AI-Glasses 边缘导航地面站 2.1 (极净多线程版) 启动...")
    if not os.path.exists(DEBUG_DIR): os.makedirs(DEBUG_DIR)
    
    cap = None
    for test_id in [0, 1, 2, 4]:
        c = cv2.VideoCapture(test_id, cv2.CAP_V4L2)
        if c.isOpened():
            ret, frame = c.read()
            if ret and frame is not None:
                cap = c
                print(f"🎯 硬件摄像头已锁死在 ID: {test_id}")
                break
            c.release()

    if cap is None:
        print("❌ 未检测到摄像头，退出。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ai_thread = threading.Thread(target=ai_inference_worker, daemon=True)
    ai_thread.start()

    print(f"📸 [巡航启航] 快门正在后台高频工作（已消音），每 {CAPTURE_INTERVAL} 秒向大模型呈递一次现状...\n")

    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if ret and frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(DEBUG_DIR, f"shot_{timestamp}.jpg")
                txt_path = os.path.join(DEBUG_DIR, f"shot_{timestamp}.txt")
                
                if image_queue.full():
                    try:
                        image_queue.get_nowait()
                        image_queue.task_done()
                    except: pass
                
                image_queue.put((frame, timestamp, img_path, txt_path))
                
                # 【修改点】用 \r 实现原地单行刷新刷新计数，不再轰炸控制台
                print(f"\r⏱️  快门心跳正常 | 刚刚捕获: shot_{timestamp}.jpg", end='', flush=True)
            
            elapsed = time.time() - start_time
            sleep_time = max(0.1, CAPTURE_INTERVAL - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n👋 收到退出指令，地面站安全关闭。")
    finally:
        cap.release()

if __name__ == '__main__':
    main()