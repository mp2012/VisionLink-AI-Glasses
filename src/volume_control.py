"""
USB 耳麦音量按钮监听器

监听 USB 耳麦上 KEY_VOLUMEUP / KEY_VOLUMEDOWN 的 HID 事件，
自动调用 amixer 调节对应 ALSA 声卡的音量。

架构：
  /dev/input/event* → HID 扫描 → evdev 读取 → amixer 调节

依赖：evdev (pip install evdev)
仅在 Jetson/Linux 平台生效，缺少 evdev 或设备时静默降级。
"""
import os
import re
import time
import logging
import subprocess
import threading
from typing import Optional

from .platform import IS_JETSON, IS_LINUX

logger = logging.getLogger(__name__)

# 音量步进（百分比，每次按键调多少）
_VOLUME_STEP = 5
# 音量范围
_VOLUME_MIN = 10
_VOLUME_MAX = 100


def _parse_card_number(audio_device: str) -> int:
    """从 ALSA 设备名提取声卡编号，如 'plughw:1,0' → 1"""
    m = re.search(r"hw:(\d+)", audio_device or "")
    return int(m.group(1)) if m else 0


def _find_mixer_control(card: int) -> Optional[str]:
    """在指定声卡上查找可用的音量控制项名称"""
    if not shutil_which("amixer"):
        return None
    try:
        result = subprocess.run(
            ["amixer", "-c", str(card), "scontrols"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None

        # 按优先级查找常见的音量控制项
        priority = [
            "Speaker",
            "PCM Playback Volume",
            "Master Playback Volume",
            "PCM",
            "Master",
            "Headphone",
            "Headphone Playback Volume",
        ]
        output = result.stdout
        found = []
        for line in output.split("\n"):
            for name in priority:
                if name in line:
                    found.append((priority.index(name), name))
        if found:
            found.sort(key=lambda x: x[0])
            best = found[0][1]
            logger.info(f"声卡{card} 音量控制项: {best}")
            return best

        # 兜底：取第一个可用的简单控制项
        m = re.search(r"Simple mixer control '([^']+)'", output)
        if m:
            logger.info(f"声卡{card} 音量控制项(兜底): {m.group(1)}")
            return m.group(1)
    except Exception as e:
        logger.warning(f"amixer scontrols 异常: {e}")
    return None


def _find_volume_input_device() -> Optional[str]:
    """扫描 /dev/input/event*，找到支持音量键的输入设备"""
    try:
        import evdev
    except ImportError:
        logger.warning("evdev 未安装，音量按钮监听不可用")
        return None

    devices = evdev.list_devices()
    if not devices:
        return None

    candidates = []
    for path in devices:
        try:
            dev = evdev.InputDevice(path)
        except Exception:
            continue
        caps_raw = dev.capabilities().get(1, [])  # EV_KEY = 1
        # evdev 默认返回 List[int]，verbose=True 时返回 List[Tuple[int, ...]]
        if caps_raw and isinstance(caps_raw[0], (list, tuple)):
            key_codes = {c[0] for c in caps_raw}
        else:
            key_codes = set(caps_raw)

        has_vol_up = evdev.ecodes.KEY_VOLUMEUP in key_codes
        has_vol_down = evdev.ecodes.KEY_VOLUMEDOWN in key_codes

        if has_vol_up or has_vol_down:
            name = dev.name.lower()
            # 排除键盘（避免普通键盘的音量键干扰）
            is_keyboard = (
                "keyboard" in name
                or "keypad" in name
                or len(key_codes) > 50  # 键盘有大量按键
            )
            if not is_keyboard:
                candidates.append((path, dev.name, has_vol_up, has_vol_down))

    if not candidates:
        logger.info("未找到支持音量键的设备（非键盘类）")
        return None

    # 优先选择同时支持 VOLUMEUP + VOLUMEDOWN 的设备
    candidates.sort(key=lambda x: (not (x[2] and x[3]), x[1]))
    path, name, _, _ = candidates[0]
    logger.info(f"音量按钮设备: {path} ({name})")
    return path


def _get_current_volume(card: int, control: str) -> Optional[int]:
    """读取当前音量百分比"""
    try:
        result = subprocess.run(
            ["amixer", "-c", str(card), "sget", control],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        # 解析: "  Front Left: Playback 128 [80%] [-16.00dB]"
        m = re.search(r"\[(\d+)%\]", result.stdout)
        if m:
            return int(m.group(1))
    except Exception as e:
        logger.warning(f"amixer sget 异常: {e}")
    return None


def _set_volume(card: int, control: str, percent: int) -> bool:
    """设置音量百分比"""
    try:
        pct = max(_VOLUME_MIN, min(_VOLUME_MAX, percent))
        result = subprocess.run(
            ["amixer", "-c", str(card), "sset", control, f"{pct}%"],
            capture_output=True, text=True, timeout=3,
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"amixer sset 异常: {e}")
        return False


def shutil_which(cmd: str) -> bool:
    """检查命令是否可用"""
    import shutil as _shutil
    return _shutil.which(cmd) is not None


# ==================== 主类 ====================

class VolumeController:
    """USB 耳麦音量按钮 → ALSA 音量调节器

    用法：
        vc = VolumeController(audio_device="plughw:1,0")
        vc.start()   # 启动后台监听线程
        ...
        vc.stop()    # 停止监听
    """

    def __init__(self, audio_device: str = None):
        self._audio_device = audio_device
        self._card = _parse_card_number(audio_device) if audio_device else 0
        self._control: Optional[str] = None
        self._input_path: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._available = False

        if not (IS_JETSON or IS_LINUX):
            logger.debug("非 Linux/Jetson 平台，音量控制器不可用")
            return

        # 查找 amixer 控制项
        if not shutil_which("amixer"):
            logger.warning("amixer 未安装，音量控制器不可用")
            return
        self._control = _find_mixer_control(self._card)
        if not self._control:
            logger.warning(f"声卡{self._card} 未找到音量控制项")
            return

        # 查找输入设备
        try:
            self._input_path = _find_volume_input_device()
        except Exception as e:
            logger.warning(f"扫描音量按钮设备异常: {e}，按钮功能不可用")
            return
        if not self._input_path:
            logger.info("未找到音量按钮设备（非键盘类），请确认 USB 耳麦已连接")
            # 不阻止启动，后续可热插拔
            return

        self._available = True
        current = _get_current_volume(self._card, self._control)
        if current is not None:
            logger.info(f"音量控制器就绪: 声卡{self._card}, 当前音量 {current}%")
        else:
            logger.info(f"音量控制器就绪: 声卡{self._card}")

    @property
    def available(self) -> bool:
        return self._available

    def start(self):
        """启动后台监视线程"""
        if not self._available or self._running:
            return
        # 重试查找输入设备（支持热插拔）
        if not self._input_path:
            self._input_path = _find_volume_input_device()
            if not self._input_path:
                logger.warning("音量按钮设备仍不可用，跳过启动")
                return

        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True, name="vol-ctrl")
        self._thread.start()
        logger.info("音量按钮监听已启动")

    def stop(self):
        """停止后台监视线程"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("音量按钮监听已停止")

    def _reconnect_input(self) -> Optional[object]:
        """尝试重新打开输入设备"""
        try:
            import evdev
            if not self._input_path or not os.path.exists(self._input_path):
                self._input_path = _find_volume_input_device()
                if not self._input_path:
                    return None
            return evdev.InputDevice(self._input_path)
        except Exception:
            self._input_path = _find_volume_input_device()
            if self._input_path:
                try:
                    import evdev
                    return evdev.InputDevice(self._input_path)
                except Exception:
                    pass
            return None

    def _listen(self):
        """后台线程：循环读取 evdev 事件"""
        import evdev

        device = None
        last_event_time = 0.0
        reopen_interval = 5.0  # 每 5 秒重试打开设备

        try:
            device = self._reconnect_input()
        except Exception:
            pass

        while self._running:
            # 如果设备不可用，定时重试
            if device is None:
                time.sleep(reopen_interval)
                try:
                    device = self._reconnect_input()
                except Exception:
                    continue
                continue

            try:
                event = device.read_one()
            except BlockingIOError:
                time.sleep(0.05)
                continue
            except Exception:
                # 设备断开
                logger.debug("音量按钮设备断开，等待重连...")
                try:
                    device.close()
                except Exception:
                    pass
                device = None
                time.sleep(reopen_interval)
                continue

            if event is None:
                time.sleep(0.05)
                continue

            # 只处理按键按下（value=1）
            if event.type != evdev.ecodes.EV_KEY or event.value != 1:
                continue

            # 防抖：200ms 内忽略重复
            now = time.monotonic()
            if now - last_event_time < 0.2:
                continue
            last_event_time = now

            if event.code == evdev.ecodes.KEY_VOLUMEUP:
                self._adjust_volume(+_VOLUME_STEP)
            elif event.code == evdev.ecodes.KEY_VOLUMEDOWN:
                self._adjust_volume(-_VOLUME_STEP)

        # 清理
        if device:
            try:
                device.close()
            except Exception:
                pass

    def _adjust_volume(self, delta: int):
        """调节音量并打日志"""
        if not self._control:
            return
        current = _get_current_volume(self._card, self._control)
        if current is None:
            logger.warning("无法读取当前音量")
            return
        new_vol = current + delta
        new_vol = max(_VOLUME_MIN, min(_VOLUME_MAX, new_vol))
        if new_vol == current:
            return
        if _set_volume(self._card, self._control, new_vol):
            logger.info(f"音量: {current}% → {new_vol}%")
        else:
            logger.warning(f"音量调节失败")
