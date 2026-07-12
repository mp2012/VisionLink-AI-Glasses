"""
统一配置中心
所有可调参数一处管理，平台差异自动切换。
支持双摄像头架构：POV（镜腿单目）+ FOV（胸前深度相机）
"""
import os
import threading
from .platform import IS_JETSON, IS_WINDOWS

# ==================== 模型配置 ====================
MODEL_NAME = "gemma4:e2b-it-qat"

# ==================== 双摄像头配置 ====================
# POV 摄像头（镜腿单目）：高清晰度，用于大模型推理
POV_CAMERA_CONFIG = {
    "cam_id": 0 if IS_JETSON else 0,
    "backend": "V4L2" if IS_JETSON else "DSHOW",
    "width": 640 if IS_JETSON else 800,
    "height": 480 if IS_JETSON else 600,
    "auto_scan_ids": [0, 1, 2, 4, 5] if IS_JETSON else None,
    "gstreamer_pipeline": True if IS_JETSON else False,
    "framerate": 30,
}

# FOV 摄像头（胸前深度相机）：常开流式，用于 YOLO 避障
FOV_CAMERA_CONFIG = {
    "cam_id": 2 if IS_JETSON else 1,
    "backend": "V4L2" if IS_JETSON else "DSHOW",
    "width": 640 if IS_JETSON else 640,
    "height": 480 if IS_JETSON else 480,
    "auto_scan_ids": [2, 1, 4, 5] if IS_JETSON else None,
    "gstreamer_pipeline": True if IS_JETSON else False,
    "framerate": 30,
}

# 向后兼容（旧代码可能使用 CAMERA_CONFIG）
CAMERA_CONFIG = POV_CAMERA_CONFIG

# ==================== AI 推理配置 ====================
# gemma4 图片 token 数约 (size/14)^2，288→约 420 tokens，224→约 256 tokens
# Jetson 内存有限，用较小尺寸避免 GGML_SCHED_MAX_SPLIT_INPUTS 超限
AI_IMAGE_SIZE = 224 if IS_JETSON else 448
JPEG_QUALITY = 70
TIMEOUT_INFER = 30 if IS_JETSON else 12
TIMEOUT_ASR = 8
AGENT_SCAN_INTERVAL = 6.0 if IS_JETSON else 5.0
BROADCAST_COOLDOWN = 15
LONG_TIME_LIMIT = 40
MAX_CONTEXT_ROUND = 3

# 推理额外选项（Jetson 限制上下文窗口）
# gemma4 是 thinking 模型，thinking 消耗 token 多
# GPU 模式下速度提升，num_predict 可适当增大
# 768 足够完成 thinking + 简短输出，场景描述约 20~30s
INFER_OPTIONS = {
    "timeout": TIMEOUT_INFER * 1000,
}
if IS_JETSON:
    INFER_OPTIONS["num_ctx"] = 1024
    INFER_OPTIONS["num_predict"] = 768

# ==================== YOLO 检测配置 ====================
YOLO_CONFIG = {
    "model_path": "yolov8n.pt",          # YOLOv8 nano 模型路径
    "confidence_threshold": 0.5,          # 检测置信度阈值
    "nms_threshold": 0.45,                # NMS 阈值
    "detect_classes": [0, 1, 2, 3, 5, 7],  # person, bicycle, car, motorcycle, bus, truck
    "detect_interval": 0.1,               # 检测间隔（秒），约 10 FPS
    "depth_warning_distance": 1.5,        # 深度预警距离（米），无深度相机时的估算值
    "depth_danger_distance": 0.5,         # 深度危险距离（米），无深度相机时的估算值
    "announce_cooldown": 2.0,             # 播报冷却时间（秒），避免重复播报
}

# ==================== Orbbec 深度相机配置 ====================
DEPTH_CONFIG = {
    "enabled": True,                      # 是否启用深度相机
    "warning_distance_mm": 1500,          # 预警距离（mm），1.5 米
    "danger_distance_mm": 500,            # 危险距离（mm），0.5 米
    "frame_timeout_ms": 50,               # 深度帧获取超时（ms），不宜太大以免阻塞 YOLO 循环
}

# ==================== 音频配置 ====================
if IS_JETSON:
    # 自动检测 AB13X USB Audio 设备号（避免 card 编号漂移）
    AUDIO_DEVICE = None
    try:
        with open("/proc/asound/cards", "r") as f:
            for line in f:
                if "AB13X" in line:
                    card_num = line.strip().split()[0]
                    AUDIO_DEVICE = f"plughw:{card_num},0"
                    break
    except Exception:
        pass
    if AUDIO_DEVICE is None:
        AUDIO_DEVICE = "plughw:0,0"  # 回退默认值

    # 离线 TTS 配置：优先 piper（音质好），回退 espeak-ng
    TTS_ENGINE = "piper"  # "piper" | "espeak" | "edge"（在线）
    TTS_PIPER_MODEL = "/opt/seeed/development_guide/12_llm_offline/seeed_ws/src/largemodel/MODELS/tts/zh/zh_CN-huayan-medium.onnx"
    TTS_PIPER_CONFIG = "/opt/seeed/development_guide/12_llm_offline/seeed_ws/src/largemodel/MODELS/tts/zh/zh_CN-huayan-medium.onnx.json"
    TTS_VOICE = "zh-CN-XiaoxiaoNeural"  # 仅 edge-tts 模式使用
else:
    AUDIO_DEVICE = None
    TTS_ENGINE = "edge"
    TTS_PIPER_MODEL = None
    TTS_PIPER_CONFIG = None
    TTS_VOICE = None

# 音效文件路径
SOUND_EFFECTS = {
    "shutter": os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "audio", "pic.mp3"),
    "beep": None,  # 动态生成
    "alert": None,
}

# ==================== UI 配置 ====================
UI_PANEL_WIDTH = 320
UI_ALPHA = 0.65
UI_YOLO_BOX_COLOR = (0, 255, 0)    # 绿色边界框
UI_DANGER_COLOR = (0, 0, 255)       # 红色危险
UI_WARNING_COLOR = (0, 255, 255)    # 黄色警告

# ==================== 空格防抖 ====================
SPACE_DEBOUNCE = 0.8  # 秒

# ==================== 字体配置 ====================
if IS_JETSON:
    FONT_PATHS = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
else:
    FONT_PATHS = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]

FONT_SIZES = (24, 22, 20) if IS_WINDOWS else (24, 22, 20)

# ==================== 目录配置 ====================
SAVE_DIR = "saved_captures"
if IS_JETSON:
    SNAPSHOT_DIR = "snapshots"
    INFER_LOG_DIR = "infer_logs"
    AUDIO_CACHE_DIR = "audio_cache"

# ==================== 状态枚举 ====================
STATE_IDLE = 0
STATE_CAPTURE = 1
STATE_LISTEN = 2
STATE_INFER = 3
STATE_TTS = 4

# ==================== 模式名称 ====================
MODE_NAMES = ["障碍物检测", "文字识别", "人脸检测", "场景描述", "图文问答"]

# ==================== 独占模式信号量 ====================
# 当 VLM 推理进行时，YOLO 避障线程暂停以释放 GPU 显存
# 推理完成后恢复 YOLO，避免 GGML_SCHED_MAX_SPLIT_INPUTS 崩溃
YOLO_PAUSE_EVENT = threading.Event()
YOLO_PAUSE_EVENT.set()  # 初始状态：未暂停（YOLO 正常运行）
