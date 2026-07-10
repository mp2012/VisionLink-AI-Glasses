#!/usr/bin/env python3
"""
摄像头诊断脚本
检查系统中所有可用的摄像头设备，并测试读取帧
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 检测平台
try:
    from src.platform import IS_JETSON, IS_WINDOWS
except ImportError:
    IS_JETSON = os.uname().machine.startswith("aarch64") if hasattr(os, 'uname') else False
    IS_WINDOWS = sys.platform == "win32"

print("=" * 60)
print(f"  摄像头诊断工具")
print(f"  平台: {'Jetson' if IS_JETSON else 'Windows' if IS_WINDOWS else 'Linux x86'}")
print("=" * 60)

# ========== 1. 检查 /dev/video* 设备 ==========
print("\n[1] 检查 V4L2 设备:")
try:
    import glob
    video_devices = sorted(glob.glob("/dev/video*"))
    if video_devices:
        for dev in video_devices:
            print(f"    ✓ 发现设备: {dev}")
    else:
        print("    ✗ 未发现 /dev/video* 设备")
except Exception as e:
    print(f"    ✗ 无法扫描 /dev/video: {e}")

# ========== 2. 使用 v4l2-ctl 检查详细信息（仅 Linux）==========
if not IS_WINDOWS:
    print("\n[2] V4L2 设备详细信息:")
    try:
        import subprocess
        for i in range(8):
            dev = f"/dev/video{i}"
            if os.path.exists(dev):
                print(f"\n  --- {dev} ---")
                # 获取设备名称
                result = subprocess.run(
                    ["v4l2-ctl", "-d", dev, "--info"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        line = line.strip()
                        if line and not line.startswith('VIDIOC'):
                            print(f"    {line}")
                else:
                    print(f"    ✗ 无法获取信息: {result.stderr.strip()}")
                
                # 获取支持的格式
                result2 = subprocess.run(
                    ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
                    capture_output=True, text=True, timeout=5
                )
                if result2.returncode == 0:
                    for line in result2.stdout.strip().split('\n'):
                        line = line.strip()
                        if line and not line.startswith('VIDIOC'):
                            print(f"    {line}")
    except FileNotFoundError:
        print("    ⚠ v4l2-ctl 未安装，跳过详细信息")
    except Exception as e:
        print(f"    ✗ 获取设备信息失败: {e}")

# ========== 3. OpenCV 摄像头扫描 ==========
print("\n[3] OpenCV 摄像头扫描（逐一测试 0-9）:")
working_cameras = []

for cam_id in range(10):
    for backend_name, backend in [("DEFAULT", None), ("V4L2", cv2.CAP_V4L2), ("GSTREAMER", cv2.CAP_GSTREAMER)]:
        if backend_name == "GSTREAMER" and not IS_JETSON:
            continue
        if backend_name == "V4L2" and IS_WINDOWS:
            continue
        try:
            if backend is not None:
                cap = cv2.VideoCapture(cam_id, backend)
            else:
                cap = cv2.VideoCapture(cam_id)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    working_cameras.append((cam_id, backend_name, actual_w, actual_h, fps, frame.shape))
                    print(f"    ✓ ID={cam_id} [{backend_name}] -> {actual_w}x{actual_h} @ {fps:.1f}FPS, shape={frame.shape}")
                    cap.release()
                    break  # 找到即可
                cap.release()
            else:
                cap.release()
        except Exception as e:
            pass

if not working_cameras:
    print("    ✗ 未发现任何可用摄像头！")

# ========== 4. 测试双摄像头配置 ==========
print("\n[4] 测试项目双摄像头配置:")
try:
    from src.camera import CameraManager, DualCameraManager
    from src.config import POV_CAMERA_CONFIG, FOV_CAMERA_CONFIG
    
    print(f"\n  POV 配置: cam_id={POV_CAMERA_CONFIG['cam_id']}, {POV_CAMERA_CONFIG['width']}x{POV_CAMERA_CONFIG['height']}")
    print(f"  FOV 配置: cam_id={FOV_CAMERA_CONFIG['cam_id']}, {FOV_CAMERA_CONFIG['width']}x{FOV_CAMERA_CONFIG['height']}")
    
    # 测试 POV
    print("\n  --- 测试 POV 摄像头 ---")
    pov = CameraManager(POV_CAMERA_CONFIG, name="POV-Test")
    if pov.open():
        ret, frame = pov.read()
        if ret and frame is not None:
            print(f"    ✓ POV 读取成功: shape={frame.shape}")
        else:
            print(f"    ✗ POV 读取帧失败")
        pov.release()
    else:
        print(f"    ✗ POV 打开失败")
    
    # 测试 FOV
    print("\n  --- 测试 FOV 摄像头 ---")
    fov = CameraManager(FOV_CAMERA_CONFIG, name="FOV-Test")
    if fov.open():
        ret, frame = fov.read()
        if ret and frame is not None:
            print(f"    ✓ FOV 读取成功: shape={frame.shape}")
        else:
            print(f"    ✗ FOV 读取帧失败")
        fov.release()
    else:
        print(f"    ✗ FOV 打开失败")
    
    # 测试双摄像头
    print("\n  --- 测试双摄像头协同 ---")
    dual = DualCameraManager()
    pov_ok, fov_ok = dual.open_both()
    print(f"    POV: {'✓' if pov_ok else '✗'} | FOV: {'✓' if fov_ok else '✗'}")
    if pov_ok:
        ret, frame = dual.read_pov()
        print(f"    POV 帧读取: {'✓' if ret else '✗'}")
    if fov_ok:
        ret, frame = dual.read_fov()
        print(f"    FOV 帧读取: {'✓' if ret else '✗'}")
    dual.release_all()

except ImportError as e:
    print(f"    ✗ 导入模块失败: {e}")
except Exception as e:
    print(f"    ✗ 测试异常: {e}")

# ========== 5. 总结 ==========
print("\n" + "=" * 60)
print("  诊断总结")
print("=" * 60)
print(f"  发现的 V4L2 设备: {len(video_devices) if 'video_devices' in dir() else 'N/A'}")
print(f"  OpenCV 可用摄像头: {len(working_cameras)} 个")
if working_cameras:
    for cam_id, backend, w, h, fps, shape in working_cameras:
        print(f"    - ID={cam_id} [{backend}]: {w}x{h}, {fps:.0f}FPS")
else:
    print(f"    ⚠ 没有发现可用摄像头，请检查:")
    print(f"      1. 摄像头是否正确连接")
    print(f"      2. USB 端口是否正常供电")
    print(f"      3. 当前用户是否有 /dev/video* 权限 (尝试 sudo)")
    print(f"      4. 是否有其他程序占用了摄像头")
print()
