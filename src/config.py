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
# 168→~144 tokens，112→~64 tokens
# Jetson VRAM 极小场景下进一步缩小防止 GGML_SCHED_MAX_SPLIT_INPUTS 崩溃
AI_IMAGE_SIZE = 168 if IS_JETSON else 448
JPEG_QUALITY = 70
TIMEOUT_INFER = 30 if IS_JETSON else 12
TIMEOUT_ASR = 8
AGENT_SCAN_INTERVAL = 6.0 if IS_JETSON else 5.0
BROADCAST_COOLDOWN = 15
LONG_TIME_LIMIT = 40
MAX_CONTEXT_ROUND = 3

# 推理额外选项
# Jetson 部分 offload 策略说明：
#   - 不设置 num_ctx，让 Ollama 使用模型默认上下文窗口
#     强制小 num_ctx 会导致 GGML 图分割过多，触发 GGML_SCHED_MAX_SPLIT_INPUTS 崩溃
#   - 设置 num_gpu=999 强制全 GPU offload，跳过 Ollama 自动"内存拟合"拆分逻辑
#     该拆分逻辑在 gemma4:e2b 上存在已知 bug（ollama/ollama#16506），
#     当可用显存不足以完整装下模型时会触发 GGML_ASSERT 崩溃整个 llama-server 子进程
INFER_OPTIONS = {
    "timeout": TIMEOUT_INFER * 1000,
}
if IS_JETSON:
    # 给 gemma4 thinking 阶段留足 token 预算，避免思维链耗尽预算导致答案截断
    INFER_OPTIONS["num_predict"] = 1024  # 限制最大生成 token 数（原512→1024）
    INFER_OPTIONS["num_gpu"] = 999       # 强制全 GPU offload，绕过 gemma4 拆分 bug
    # 尝试关闭思维链模式：若当前 Ollama 版本支持，可同时提速并避免截断问题
    # 若不支持则 Ollama 会静默忽略此参数
    INFER_OPTIONS["think"] = False

# ==================== YOLO 检测配置 ====================
# TensorRT 引擎路径（Jetson 专用，运行 scripts/export_yolo_trt.py 导出）
# 引擎文件与 GPU 架构绑定，更换设备或更新 JetPack 后需重新导出
# 删除 .engine 文件即自动回退到 PyTorch 模式
YOLO_ENGINE_PATH = "yolov8n.engine"

YOLO_CONFIG = {
    "model_path": "yolov8n.pt",          # YOLOv8 nano 模型路径
    "engine_path": YOLO_ENGINE_PATH,     # TensorRT FP16 引擎路径（Jetson 优先加载）
    "confidence_threshold": 0.5,          # 检测置信度阈值
    "nms_threshold": 0.45,                # NMS 阈值
    "detect_classes": [0, 1, 2, 3, 5, 7],  # person, bicycle, car, motorcycle, bus, truck
    "detect_interval": 0.05,              # 检测间隔（秒），TensorRT 下约 20 FPS（PyTorch 时约为 10 FPS）
    "depth_warning_distance": 1.5,        # 深度预警距离（米），无深度相机时的估算值
    "depth_danger_distance": 0.5,         # 深度危险距离（米），无深度相机时的估算值
    "announce_cooldown": 2.0,             # 播报冷却时间（秒），避免重复播报
}

# ==================== Orbbec 深度相机配置 ====================
# 限制 SDK 内部帧缓冲池上限（MB），双保险防止 2GB OOM
# 注意：OrbbecSDKConfig.xml 中去掉了内存池，此环境变量为兜底
os.environ.setdefault("OB_MEMORY_LIMIT_MB", "256")

DEPTH_CONFIG = {
    "enabled": True,                      # 是否启用深度相机
    "warning_distance_mm": 1500,          # 预警距离（mm），1.5 米
    "danger_distance_mm": 500,            # 危险距离（mm），0.5 米
    "frame_timeout_ms": 50,               # 深度帧获取超时（ms），不宜太大以免阻塞 YOLO 循环
}

# ==================== Web 预览配置 ====================
# 局域网内通过浏览器实时预览深度相机画面
WEB_PREVIEW_PORT = 5000
WEB_PREVIEW_JPEG_QUALITY = 80

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
    TTS_PIPER_MODEL = os.environ.get(
        "VISIONLINK_TTS_MODEL",
        "/opt/seeed/development_guide/12_llm_offline/seeed_ws/src/largemodel/MODELS/tts/zh/zh_CN-huayan-medium.onnx",
    )
    TTS_PIPER_CONFIG = os.environ.get(
        "VISIONLINK_TTS_CONFIG",
        "/opt/seeed/development_guide/12_llm_offline/seeed_ws/src/largemodel/MODELS/tts/zh/zh_CN-huayan-medium.onnx.json",
    )
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

FONT_SIZES = (24, 22, 20)

# ==================== 目录配置 ====================
if IS_JETSON:
    SNAPSHOT_DIR = "snapshots"
    INFER_LOG_DIR = "infer_logs"
    AUDIO_CACHE_DIR = "audio_cache"
else:
    SNAPSHOT_DIR = "snapshots"
    INFER_LOG_DIR = "infer_logs"
    AUDIO_CACHE_DIR = "audio_cache"

# ==================== 状态枚举 ====================
STATE_IDLE = 0
STATE_INFER = 3
STATE_TTS = 4

# ==================== 模式名称 ====================
MODE_NAMES = ["障碍物检测", "文字识别", "人脸检测", "场景描述", "图文问答"]

# ==================== TTS 优先级（数字越小优先级越高） ====================
# 紧急避障 > 普通预警 > 系统提示 > 常规播报
# 高优先级可以打断低优先级，低优先级不允许打断高优先级
# 特殊规则：同级 P0 允许互相打断（新危险信息替换旧危险信息）
TTS_PRIORITY_EMERGENCY = 0   # 紧急避障（danger）：直接威胁人身安全，必须立即打断
TTS_PRIORITY_WARNING = 1     # 普通预警（warning）：潜在障碍，不应被 VLM 播报阻塞
TTS_PRIORITY_SYSTEM = 2      # 系统提示：模式切换、开关语音等
TTS_PRIORITY_NORMAL = 3      # 常规播报：VLM 推理结果、场景描述等

# P0 危险警报过时阈值（秒）：超过此时间的警报不再播放
# 理由：危险警报的实时性要求极高，过时信息会误导用户做出错误判断
TTS_ALERT_STALE_THRESHOLD = 2.5  # 秒

# ==================== 独占模式信号量 ====================
# 当 VLM 推理进行时，YOLO 避障线程暂停以释放 GPU 显存
# 推理完成后恢复 YOLO，避免 GGML_SCHED_MAX_SPLIT_INPUTS 崩溃
YOLO_PAUSE_EVENT = threading.Event()
YOLO_PAUSE_EVENT.set()  # 初始状态：未暂停（YOLO 正常运行）
