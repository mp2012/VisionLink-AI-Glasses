#!/usr/bin/env python3
"""
VisionLink 系统综合诊断脚本
一键检查所有硬件、软件、模块是否正常
"""
import sys
import os
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

errors = []
warnings = []

def check(label, condition, detail="", error_msg=""):
    status = PASS if condition else FAIL
    msg = f"  {status} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    if not condition and error_msg:
        errors.append(error_msg or label)

def check_warn(label, condition, detail=""):
    status = PASS if condition else WARN
    msg = f"  {status} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    if not condition:
        warnings.append(label)

print("=" * 60)
print("  VisionLink 系统综合诊断")
print("=" * 60)

# ==================== 1. 平台信息 ====================
print("\n[1] 平台信息")
try:
    from src.platform import IS_JETSON, IS_WINDOWS
    platform = "Jetson" if IS_JETSON else "Windows" if IS_WINDOWS else "Linux x86"
    print(f"  平台: {platform}")
except ImportError as e:
    print(f"  {FAIL} 平台检测失败: {e}")
    errors.append("无法导入 platform 模块")
    IS_JETSON = False

# ==================== 2. 核心依赖 ====================
print("\n[2] 核心依赖")

# Python
import sys
py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
check(f"Python {py_ver}", sys.version_info >= (3, 9))

# OpenCV
try:
    import cv2
    check("OpenCV", True, cv2.__version__)
except ImportError:
    check("OpenCV", False, error_msg="OpenCV 未安装")

# NumPy
try:
    import numpy as np
    check("NumPy", True, np.__version__)
except ImportError:
    check("NumPy", False, error_msg="NumPy 未安装")

# Ollama
try:
    import ollama
    ver = getattr(ollama, "__version__", "已安装")
    check("Ollama 客户端", True, ver)
except ImportError:
    check("Ollama 客户端", False, error_msg="ollama 包未安装")

# edge-tts
result = subprocess.run(["pip3", "show", "edge-tts"], capture_output=True, text=True)
if result.returncode == 0:
    ver = [l.split(":")[1].strip() for l in result.stdout.split("\n") if l.startswith("Version:")]
    check("edge-tts", True, ver[0] if ver else "已安装")
else:
    check("edge-tts", False, error_msg="edge-tts 未安装")

# ultralytics (YOLO)
result = subprocess.run(["pip3", "show", "ultralytics"], capture_output=True, text=True)
if result.returncode == 0:
    ver = [l.split(":")[1].strip() for l in result.stdout.split("\n") if l.startswith("Version:")]
    check("ultralytics (YOLO)", True, ver[0] if ver else "已安装")
else:
    check("ultralytics (YOLO)", False, error_msg="ultralytics 未安装")

# ffmpeg/ffplay
for tool in ["ffmpeg", "ffplay"]:
    result = subprocess.run(["which", tool], capture_output=True, text=True)
    check(tool, result.returncode == 0, result.stdout.strip() if result.returncode == 0 else "")

# sox
result = subprocess.run(["which", "sox"], capture_output=True, text=True)
check("sox", result.returncode == 0, result.stdout.strip() if result.returncode == 0 else "")

# ==================== 3. Ollama 服务 ====================
print("\n[3] Ollama 推理服务")
try:
    import requests
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    from src.config import MODEL_NAME
    check("Ollama 服务运行中", True)
    check(f"模型 {MODEL_NAME} 已加载", MODEL_NAME in models,
          f"可用模型: {', '.join(models)}" if models else "无模型")
except Exception as e:
    check("Ollama 服务", False, str(e)[:60], error_msg=f"Ollama 不可达: {e}")

# ==================== 4. YOLO 模型文件 ====================
print("\n[4] YOLO 模型")
from src.config import YOLO_CONFIG
yolo_path = YOLO_CONFIG["model_path"]
if os.path.exists(yolo_path):
    size_mb = os.path.getsize(yolo_path) / (1024 * 1024)
    check(f"YOLO 模型文件 {yolo_path}", True, f"{size_mb:.1f}MB")
else:
    check(f"YOLO 模型文件 {yolo_path}", False, error_msg=f"YOLO 模型文件缺失: {yolo_path}")

# 尝试加载模型
try:
    from ultralytics import YOLO
    model = YOLO(yolo_path)
    check("YOLO 模型加载", True)
except Exception as e:
    check("YOLO 模型加载", False, str(e)[:60], error_msg=f"YOLO 加载失败: {e}")

# ==================== 5. 摄像头 ====================
print("\n[5] 摄像头")

# 检查 /dev/video* 设备
import glob
video_devs = sorted(glob.glob("/dev/video*"))
if video_devs:
    print(f"  发现 {len(video_devs)} 个 V4L2 设备: {', '.join(video_devs)}")
else:
    check("V4L2 设备", False, error_msg="未发现 /dev/video* 设备")

# 检查是否有进程占用摄像头
result = subprocess.run(["fuser"] + video_devs, capture_output=True, text=True, timeout=3)
if result.stdout.strip():
    pids = set()
    for line in result.stdout.strip().split('\n'):
        for part in line.split()[1:]:
            pids.add(part.rstrip('m'))
    print(f"  {WARN} 摄像头被进程占用: PID={', '.join(sorted(pids))}")
    for pid in sorted(pids):
        p = subprocess.run(["ps", "-p", pid, "-o", "pid,cmd", "--no-headers"],
                          capture_output=True, text=True)
        print(f"     {p.stdout.strip()}")
else:
    print(f"  {PASS} 摄像头未被占用")

from src.config import POV_CAMERA_CONFIG, FOV_CAMERA_CONFIG

try:
    from src.camera import CameraManager, DualCameraManager

    # POV
    pov = CameraManager(POV_CAMERA_CONFIG, name="POV")
    pov_ok = pov.open()
    if pov_ok:
        ret, frame = pov.read()
        if ret and frame is not None:
            check("POV 摄像头", True, f"ID={POV_CAMERA_CONFIG['cam_id']} {frame.shape[1]}x{frame.shape[0]}")
        else:
            check("POV 摄像头", False, "打开成功但读帧失败")
        pov.release()
    else:
        check("POV 摄像头", False, f"ID={POV_CAMERA_CONFIG['cam_id']} 打开失败")

    # FOV
    fov = CameraManager(FOV_CAMERA_CONFIG, name="FOV")
    fov_ok = fov.open()
    if fov_ok:
        ret, frame = fov.read()
        if ret and frame is not None:
            check("FOV 摄像头", True, f"ID={FOV_CAMERA_CONFIG['cam_id']} {frame.shape[1]}x{frame.shape[0]}")
        else:
            check("FOV 摄像头", False, "打开成功但读帧失败")
        fov.release()
    else:
        check("FOV 摄像头", False, f"ID={FOV_CAMERA_CONFIG['cam_id']} 打开失败")

    # 双摄像头
    if pov_ok and fov_ok:
        dual = DualCameraManager()
        dp, df = dual.open_both()
        if dp and df:
            rp, _ = dual.read_pov()
            rf, _ = dual.read_fov()
            check("双摄像头协同", rp and rf,
                  "同时读取" if rp and rf else "读取失败")
        else:
            check("双摄像头协同", False, f"POV={'OK' if dp else 'FAIL'} FOV={'OK' if df else 'FAIL'}")
        dual.release_all()

except Exception as e:
    check("摄像头模块", False, str(e)[:60], error_msg=f"摄像头测试异常: {e}")

# ==================== 6. 音频 ====================
print("\n[6] 音频")

from src.config import AUDIO_DEVICE
check("音频设备配置", AUDIO_DEVICE is not None, AUDIO_DEVICE or "未配置")

if AUDIO_DEVICE:
    # 测试 aplay
    gen = subprocess.run(
        ["sox", "-n", "-r", "22050", "-c", "1", "/tmp/_diag_beep.wav",
         "synth", "0.1", "sine", "800", "vol", "0.3"],
        capture_output=True, text=True, timeout=5
    )
    if gen.returncode == 0:
        play = subprocess.run(
            ["aplay", "-D", AUDIO_DEVICE, "/tmp/_diag_beep.wav"],
            capture_output=True, text=True, timeout=3
        )
        check(f"ALSA 播放 ({AUDIO_DEVICE})", play.returncode == 0,
              "" if play.returncode == 0 else play.stderr.strip()[:50])
    else:
        check("beep 生成", False, gen.stderr.strip()[:50])

# 音效文件
shutter_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "audio", "pic.mp3"
)
check("快门音效文件", os.path.exists(shutter_path),
      f"{os.path.getsize(shutter_path)} bytes" if os.path.exists(shutter_path) else "")

# 离线 TTS 测试（piper > espeak-ng > edge-tts）
from src.config import TTS_ENGINE, TTS_PIPER_MODEL, TTS_PIPER_CONFIG

check("TTS 引擎配置", True, TTS_ENGINE)

if TTS_ENGINE == "piper":
    # 检查 piper 程序
    piper_bin = subprocess.run(["which", "piper"], capture_output=True, text=True)
    check("piper 程序", piper_bin.returncode == 0,
          piper_bin.stdout.strip() if piper_bin.returncode == 0 else "未找到")
    # 检查 piper 模型
    check("piper 模型", os.path.exists(TTS_PIPER_MODEL),
          f"{os.path.getsize(TTS_PIPER_MODEL)//1024//1024}MB" if os.path.exists(TTS_PIPER_MODEL) else "缺失")
    # 测试 piper 合成
    if piper_bin.returncode == 0 and os.path.exists(TTS_PIPER_MODEL):
        try:
            r = subprocess.run(
                f"echo '测试' | piper -m {TTS_PIPER_MODEL} --config {TTS_PIPER_CONFIG} "
                f"--output-raw 2>/dev/null | ffmpeg -f s16le -ar 22050 -ac 1 -i - "
                f"-f wav /tmp/_diag_piper.wav -y 2>/dev/null",
                shell=True, executable="/bin/bash", timeout=15
            )
            ok = r.returncode == 0 and os.path.exists("/tmp/_diag_piper.wav")
            check("piper 合成测试", ok,
                  f"{os.path.getsize('/tmp/_diag_piper.wav')} bytes" if ok else "失败")
            if ok:
                os.remove("/tmp/_diag_piper.wav")
        except Exception as e:
            check("piper 合成测试", False, str(e)[:50])

elif TTS_ENGINE == "espeak":
    espeak_bin = subprocess.run(["which", "espeak-ng"], capture_output=True, text=True)
    check("espeak-ng 程序", espeak_bin.returncode == 0,
          espeak_bin.stdout.strip() if espeak_bin.returncode == 0 else "未找到")
    if espeak_bin.returncode == 0:
        try:
            r = subprocess.run(
                ["espeak-ng", "-v", "zh-cmn", "测试", "-w", "/tmp/_diag_espeak.wav", "-s", "160"],
                capture_output=True, text=True, timeout=10
            )
            ok = r.returncode == 0 and os.path.exists("/tmp/_diag_espeak.wav")
            check("espeak-ng 合成测试", ok,
                  f"{os.path.getsize('/tmp/_diag_espeak.wav')} bytes" if ok else "失败")
            if ok:
                os.remove("/tmp/_diag_espeak.wav")
        except Exception as e:
            check("espeak-ng 合成测试", False, str(e)[:50])

else:
    # edge-tts 在线模式
    check("TTS 模式", True, "edge-tts (在线)")
    try:
        result = subprocess.run(
            ["python3", "-m", "edge_tts", "--voice", "zh-CN-XiaoxiaoNeural",
             "--text", "测试", "--write-media", "/tmp/_diag_tts.mp3"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists("/tmp/_diag_tts.mp3"):
            check("edge-tts 合成测试", True, f"{os.path.getsize('/tmp/_diag_tts.mp3')} bytes")
            os.remove("/tmp/_diag_tts.mp3")
        else:
            check("edge-tts 合成测试", False, result.stderr.strip()[:50])
    except Exception as e:
        check("edge-tts 合成测试", False, str(e)[:50])

# ==================== 7. 核心模块导入 ====================
print("\n[7] 核心模块")
modules = [
    ("src.config", "配置中心"),
    ("src.camera", "摄像头管理"),
    ("src.tts", "语音引擎"),
    ("src.inference", "推理引擎"),
    ("src.detection", "YOLO 检测"),
    ("src.agent", "控制中枢"),
    ("src.ui", "UI 界面"),
    ("src.prompts", "提示词"),
]
for mod_name, desc in modules:
    try:
        __import__(mod_name)
        check(desc, True)
    except Exception as e:
        check(desc, False, str(e)[:60])

# ==================== 8. 字体 ====================
print("\n[8] 中文字体")
from src.config import FONT_PATHS
font_found = None
for fp in FONT_PATHS:
    if os.path.exists(fp):
        font_found = fp
        break
check("中文字体", font_found is not None,
      font_found or "未找到",
      error_msg="中文字体缺失，UI 中文将无法渲染" if not font_found else "")

# ==================== 总结 ====================
print("\n" + "=" * 60)
print("  诊断总结")
print("=" * 60)

if errors:
    print(f"\n  {FAIL} 发现 {len(errors)} 个错误:")
    for e in errors:
        print(f"     - {e}")

if warnings:
    print(f"\n  {WARN} 发现 {len(warnings)} 个警告:")
    for w in warnings:
        print(f"     - {w}")

if not errors:
    if warnings:
        print(f"\n  {PASS} 系统基本正常，有 {len(warnings)} 个警告项")
    else:
        print(f"\n  {PASS} 系统一切正常！可以启动")
        print()
        print("  启动命令:")
        print("    python3 apps/headless.py --dual --yolo")

print()
