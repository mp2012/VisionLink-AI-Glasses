"""
统一配置中心
所有可调参数一处管理，平台差异自动切换。
"""
from .platform import IS_JETSON, IS_WINDOWS

# ==================== 模型配置 ====================
MODEL_NAME = "gemma4:e2b-q4_K_S" if IS_JETSON else "gemma4:e2b-it-qat"

# ==================== 摄像头配置 ====================
CAMERA_CONFIG = {
    "cam_id": 0 if IS_JETSON else 0,
    "backend": "V4L2" if IS_JETSON else "DSHOW",
    "width": 640 if IS_JETSON else 800,
    "height": 480 if IS_JETSON else 600,
    "auto_scan_ids": [0, 1, 2, 4, 5] if IS_JETSON else None,
}

# ==================== AI 推理配置 ====================
AI_IMAGE_SIZE = 288 if IS_JETSON else 448
JPEG_QUALITY = 70
TIMEOUT_INFER = 8 if IS_JETSON else 12
TIMEOUT_ASR = 8
AGENT_SCAN_INTERVAL = 6.0 if IS_JETSON else 5.0
BROADCAST_COOLDOWN = 15
LONG_TIME_LIMIT = 40
MAX_CONTEXT_ROUND = 3

# 推理额外选项（Jetson 限制上下文窗口）
INFER_OPTIONS = {
    "timeout": TIMEOUT_INFER * 1000,
}
if IS_JETSON:
    INFER_OPTIONS["num_ctx"] = 512
    INFER_OPTIONS["num_predict"] = 256

# ==================== UI 配置 ====================
UI_PANEL_WIDTH = 320
UI_ALPHA = 0.65

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
