"""
VisionLink 边缘端无头模式入口
运行于 Jetson Orin Nano，双摄像头协同 + YOLO 避障 + Orbbec 深度相机 + VLM 推理

启动方式:
    python apps/headless.py                    # 仅 POV 摄像头
    python apps/headless.py --dual             # 双摄像头模式（POV + FOV）
    python apps/headless.py --yolo             # 启用 YOLO 避障
    python apps/headless.py --dual --yolo      # 全功能模式（双摄 + YOLO）
    python apps/headless.py --dual --yolo --depth  # 全功能 + Orbbec 真实深度
"""
import os
import sys
import time
import logging
import argparse
import threading
import select
import termios
import tty

import cv2

# 屏蔽 Qt 字体警告
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu/")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

# 在文件描述符层面屏蔽 stderr（fd 2），对 C 扩展也有效
# 消除 OpenCV libjpeg "Corrupt JPEG data" 等噪音
# Python logging 走 stdout（fd 1），不受影响
# 注意: Orbbec SDK 的日志通过 OrbbecSDKConfig.xml (ConsoleLogLevel=5) 控制，
# 不依赖此处的 stderr 重定向
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.platform import IS_JETSON, HAS_DISPLAY
from src.config import (
    MODE_NAMES, SNAPSHOT_DIR, INFER_LOG_DIR, AUDIO_CACHE_DIR,
    YOLO_CONFIG, DEPTH_CONFIG, STATE_IDLE,
)
from src.camera import CameraManager, DualCameraManager
from src.inference import InferenceEngine
from src.tts import TTSEngine
from src.agent import Agent
from src.ui import UIManager
from src.detection import YOLODetector

# ==================== 日志配置 ====================
sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="{asctime} | {levelname:8} | {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("headless")


# ==================== 全局键盘监听（evdev） ====================
# 在无头模式下，无论光标在哪里，按键都能被捕获
KEY_MAP = {
    "KEY_SPACE": " ",
    "KEY_1": "1", "KEY_2": "2", "KEY_3": "3", "KEY_4": "4", "KEY_5": "5",
    "KEY_Y": "y", "KEY_S": "s", "KEY_Q": "q",
    "KEY_ESC": "\x1b",
}


def find_keyboard_device():
    """自动查找键盘设备（优先按键最多的输入设备）"""
    try:
        import evdev
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        # 先过滤出真正的键盘设备（排除 Consumer Control / System Control / 摄像头按钮等）
        candidates = []
        for d in devices:
            name_lower = d.name.lower()
            caps = d.capabilities().get(1, [])
            # 跳过明显的非键盘设备
            if any(skip in name_lower for skip in ["consumer control", "system control", "camera", "mouse"]):
                continue
            if len(caps) > 50:  # 真正的键盘至少 50+ 按键
                candidates.append((d.path, len(caps), d.name))

        # 按按键数量降序排列，选最多的
        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidates:
            path, count, name = candidates[0]
            logger.info(f"全局键盘设备: {path} ({name}, {count}键)")
            return path

        logger.warning("未找到合适的键盘设备")
        return None
    except ImportError:
        logger.warning("evdev 未安装，回退到终端键盘监听")
        return None


class GlobalKeyboard:
    """全局键盘监听器，使用 evdev 读取 /dev/input/event* 设备"""

    def __init__(self, device_path: str = None):
        self._device_path = device_path
        self._device = None
        self._pending_key = None

    def open(self) -> bool:
        if not self._device_path:
            self._device_path = find_keyboard_device()
        if not self._device_path:
            return False
        try:
            import evdev
            self._device = evdev.InputDevice(self._device_path)
            self._device.grab()  # 独占设备，防止按键被其他程序处理
            logger.info(f"全局键盘监听已启动: {self._device_path}")
            return True
        except Exception as e:
            logger.warning(f"全局键盘设备打开失败: {e}")
            return False

    def read_key(self) -> str:
        """非阻塞读取按键，返回按键字符或空字符串"""
        if self._device is None:
            return ""
        try:
            import evdev
            event = self._device.read_one()
            if event is None:
                return ""
            # 仅处理按键按下事件（value=1）
            if event.type == evdev.ecodes.EV_KEY and event.value == 1:
                code = evdev.ecodes.KEY.get(event.code, "")
                return KEY_MAP.get(code, "")
            return ""
        except BlockingIOError:
            return ""
        except Exception:
            return ""

    def close(self):
        if self._device:
            try:
                self._device.ungrab()
                self._device.close()
            except Exception:
                pass
            self._device = None


# 终端键盘回退（当 evdev 不可用时）
def get_terminal_key():
    """非阻塞读取终端按键"""
    if select.select([sys.stdin], [], [], 0.02)[0]:
        return sys.stdin.read(1)
    return ""


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="VisionLink 边缘端无头模式")
    parser.add_argument("--dual", action="store_true", help="启用双摄像头模式 (POV + FOV)")
    parser.add_argument("--yolo", action="store_true", help="启用 YOLO 避障检测（需要 FOV 摄像头）")
    parser.add_argument("--depth", action="store_true", help="启用 Orbbec 深度相机真实距离测量（需要 --yolo）")
    parser.add_argument("--gui", action="store_true", help="强制启用 GUI 调试窗口")
    parser.add_argument("--model", type=str, default=None, help="指定模型名称")
    args = parser.parse_args()

    # ==================== 启动日志 ====================
    logger.info("=" * 55)
    logger.info("VisionLink 边缘端启动")
    logger.info(f"平台: {'Jetson Orin Nano' if IS_JETSON else 'Linux Desktop'}")
    depth_label = " + Orbbec深度" if args.depth else ""
    logger.info(f"模式: {'双摄像头' if args.dual else '单摄像头(POV)'}"
                f"{' + YOLO避障' if args.yolo else ''}{depth_label}")
    logger.info(f"模型: {args.model or '默认'}")

    for i, name in enumerate(MODE_NAMES, 1):
        logger.info(f"  模式{i}: {name}")

    logger.info("操作: 空格=触发 | 1~5=切换模式 | Y=开关YOLO | S=停止语音 | Q=退出")
    logger.info(f"存储: {SNAPSHOT_DIR} | {INFER_LOG_DIR} | {AUDIO_CACHE_DIR}")
    logger.info("=" * 55)

    # ==================== 引擎初始化 ====================
    infer = InferenceEngine(model_name=args.model)
    tts = TTSEngine()
    ui = UIManager(enable_gui=args.gui)

    # 摄像头初始化
    if args.dual:
        camera = DualCameraManager()
        pov_ok, fov_ok = camera.open_both()
        if not pov_ok:
            logger.error("POV 摄像头初始化失败，退出")
            return
        if not fov_ok and args.yolo:
            logger.warning("FOV 摄像头不可用，YOLO 避障将被禁用")
            args.yolo = False
    else:
        camera = CameraManager(name="POV-镜腿单目")
        if not camera.open():
            logger.error("摄像头初始化失败，退出")
            return

    # 深度相机初始化
    depth_camera = None
    if args.depth and args.yolo:
        try:
            from src.orbbec_depth import OrbbecDepthCamera, is_available
            if is_available():
                depth_camera = OrbbecDepthCamera()
                if depth_camera.connect():
                    logger.info("Orbbec 深度相机已启用，YOLO 避障将使用真实距离")
                else:
                    logger.warning("Orbbec 深度相机连接失败，回退到位置估算模式")
                    depth_camera = None
            else:
                logger.warning("未检测到 Orbbec SDK，回退到位置估算模式")
        except Exception as e:
            logger.warning(f"深度相机初始化异常: {e}，回退到位置估算模式")
    elif args.depth and not args.yolo:
        logger.warning("--depth 需要配合 --yolo 使用，深度相机未启用")

    # Agent 初始化
    agent = Agent(infer, tts, camera)
    agent._yolo_enabled = args.yolo

    # YOLO 检测器初始化
    yolo_detector = None
    if args.yolo and args.dual:
        yolo_detector = YOLODetector(
            camera.fov,
            depth_camera=depth_camera,
            on_detect=agent.on_yolo_detect
        )
        if not yolo_detector.start():
            logger.warning("YOLO 检测器启动失败，避障功能不可用")
            yolo_detector = None
    elif args.yolo and not args.dual:
        logger.warning("YOLO 需要双摄像头模式，请添加 --dual 参数")
        args.yolo = False

    # 启动语音
    tts.speak("项目启动")

    # ==================== 全局键盘设置 ====================
    keyboard = GlobalKeyboard()
    if not keyboard.open():
        logger.warning("全局键盘不可用，尝试终端回退...")
        use_terminal = sys.stdin.isatty()
        if use_terminal:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            logger.info("终端键盘监听已启用（回退模式）")
        else:
            old_settings = None
            logger.warning("键盘输入完全不可用")
    else:
        use_terminal = False
        old_settings = None

    # ==================== GUI 窗口 ====================
    if args.gui:
        ui.create_debug_window("VisionLink - POV")
        if args.dual:
            ui.create_debug_window("VisionLink - FOV (YOLO)")

    # ==================== 主循环 ====================
    try:
        while True:
            # 读取 POV 帧
            if args.dual:
                pov_ret, pov_frame = camera.read_pov()
                fov_ret, fov_frame = camera.read_fov()
            else:
                pov_ret, pov_frame = camera.read()
                fov_ret, fov_frame = False, None

            if not pov_ret or pov_frame is None:
                time.sleep(0.05)
                continue

            now = time.time()

            # 自动模式扫描
            if agent.should_scan(now):
                threading.Thread(
                    target=agent.auto_scan,
                    args=(pov_frame.copy(),),
                    daemon=True
                ).start()

            # 键盘输入（优先全局 evdev，回退终端 stdin）
            if not use_terminal:
                key = keyboard.read_key()
            else:
                key = get_terminal_key()

            if not key:
                # 渲染 UI（如果启用 GUI）
                if args.gui:
                    pov_display = ui.render(
                        pov_frame, agent.state, agent.current_mode,
                        auto_enabled=agent.auto_enabled
                    )
                    ui.show_frame("VisionLink - POV", pov_display)

                    if args.dual and fov_ret and fov_frame is not None:
                        fov_display = ui.render_fov_debug(fov_frame, yolo_detector)
                        ui.show_frame("VisionLink - FOV (YOLO)", fov_display)

                    cv2.waitKey(1)
                continue

            # 按键提示音
            tts.play_beep()

            # ---- 空格键：手动触发推理 ----
            if key == " ":
                agent.handle_trigger(pov_frame)
                continue

            # ---- 1~5：切换模式 ----
            elif key in "12345":
                agent.set_mode(int(key))
                continue

            # ---- Y/y：开关 YOLO 避障 ----
            elif key.lower() == "y":
                if yolo_detector is not None:
                    agent.toggle_yolo()
                else:
                    tts.speak("YOLO 避障未启用，请使用 --dual --yolo 启动")
                continue

            # ---- S/s：停止语音 ----
            elif key.lower() == "s":
                tts.stop()
                continue

            # ---- Q/q：退出 ----
            elif key.lower() == "q":
                break

    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        # 清理
        keyboard.close()
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        if yolo_detector:
            yolo_detector.stop()

        if depth_camera:
            depth_camera.stop()

        if args.dual:
            camera.release_all()
        else:
            camera.release()

        ui.destroy()
        tts.stop()
        agent.shutdown()
        logger.info("程序结束")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"全局异常: {e}")
        raise
