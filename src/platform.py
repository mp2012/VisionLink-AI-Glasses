"""
平台检测与环境适配（Single Source of Truth）
所有模块通过本文件获取当前运行环境，避免重复检测。
"""
import os
import platform


def _check_jetson() -> bool:
    """通过设备树模型信息判定是否为 NVIDIA Jetson 硬件"""
    if platform.system() != "Linux":
        return False
    model_path = "/proc/device-tree/model"
    if os.path.exists(model_path):
        try:
            with open(model_path, "r") as f:
                model_info = f.read().lower()
                if "nvidia" in model_info or "jetson" in model_info:
                    return True
        except Exception:
            pass
    # 保底策略：aarch64 架构（可能误判树莓派 5 等非 Jetson ARM64 设备）
    if platform.machine() == "aarch64":
        import logging
        logging.getLogger(__name__).warning(
            "无法读取设备型号，通过 aarch64 架构推测为 Jetson 平台。"
            "若实际为非 Jetson ARM64 设备，请设置 VISIONLINK_PLATFORM=linux 环境变量。"
        )
        return True
    return False


IS_JETSON = _check_jetson()
IS_WINDOWS = (platform.system() == "Windows")
IS_LINUX = (platform.system() == "Linux" and not IS_JETSON)
HAS_DISPLAY = bool(os.environ.get("DISPLAY")) if not IS_WINDOWS else True


def get_platform_name() -> str:
    if IS_JETSON:
        return "Jetson Orin Nano"
    elif IS_WINDOWS:
        return "Windows Desktop"
    elif IS_LINUX:
        return "Linux Desktop"
    return "Unknown"
