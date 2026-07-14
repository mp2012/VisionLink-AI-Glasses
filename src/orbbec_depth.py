"""
Orbbec Astra Pro Plus 深度相机模块 (ctypes 封装 C++ SDK v1.10.27)

通过 ctypes 直接调用 libOrbbecSDK.so 的 C API，
读取深度图和彩色图，提供真实深度距离数据。

注意: Astra Pro Plus 的深度传感器在 SDK 中注册为 IR sensor (type=3)，
需要用 OB_SENSOR_IR 来获取 stream profile。

依赖: Orbbec C++ SDK v1.10.27 需安装在 ~/.local/lib/
"""

import ctypes
import numpy as np
import time
import os
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================
OB_SENSOR_DEPTH = 1
OB_SENSOR_COLOR = 2
OB_SENSOR_IR = 3

OB_PROFILE_DEFAULT = 0

# ============================================================
# 加载 SDK 库（自动查找已安装路径）
# ============================================================
_LIB = None


def _find_lib():
    """自动查找 libOrbbecSDK.so 路径"""
    search_paths = [
        os.path.expanduser("~/.local/lib"),
        "/usr/local/lib",
        "/usr/lib",
    ]
    for base in search_paths:
        for name in ["libOrbbecSDK.so.1.10.27", "libOrbbecSDK.so.1.10", "libOrbbecSDK.so"]:
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
    return None


def is_available() -> bool:
    """检查深度相机 SDK 是否可用"""
    return _find_lib() is not None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    lib_path = _find_lib()
    if lib_path is None:
        raise RuntimeError(
            "未找到 libOrbbecSDK.so。请安装 Orbbec C++ SDK v1.10.27: "
            "https://github.com/orbbec/OrbbecSDK"
        )

    lib_dir = os.path.dirname(lib_path)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}"

    _LIB = ctypes.CDLL(lib_path)
    _setup_api(_LIB)
    return _LIB


def _setup_api(lib):
    """设置 ctypes 函数签名"""
    _err_t = ctypes.POINTER(ctypes.c_void_p)

    # --- Context ---
    lib.ob_create_context.restype = ctypes.c_void_p
    lib.ob_delete_context.restype = None
    lib.ob_delete_context.argtypes = [ctypes.c_void_p]

    # --- Device info ---
    lib.ob_query_device_list.restype = ctypes.c_void_p
    lib.ob_query_device_list.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_delete_device_list.restype = None
    lib.ob_delete_device_list.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_device_list_device_count.restype = ctypes.c_int
    lib.ob_device_list_device_count.argtypes = [ctypes.c_void_p]
    lib.ob_device_list_get_device_name.restype = ctypes.c_char_p
    lib.ob_device_list_get_device_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ob_device_list_get_device_pid.restype = ctypes.c_int
    lib.ob_device_list_get_device_pid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ob_device_list_get_device_serial_number.restype = ctypes.c_char_p
    lib.ob_device_list_get_device_serial_number.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # --- Pipeline ---
    lib.ob_create_pipeline.restype = ctypes.c_void_p
    lib.ob_create_pipeline.argtypes = [_err_t]
    lib.ob_delete_pipeline.restype = None
    lib.ob_delete_pipeline.argtypes = [ctypes.c_void_p]
    lib.ob_pipeline_start_with_config.restype = None
    lib.ob_pipeline_start_with_config.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _err_t]
    lib.ob_pipeline_stop.restype = None
    lib.ob_pipeline_stop.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_pipeline_wait_for_frameset.restype = ctypes.c_void_p
    lib.ob_pipeline_wait_for_frameset.argtypes = [ctypes.c_void_p, ctypes.c_int, _err_t]

    # --- Stream profiles ---
    lib.ob_pipeline_get_stream_profile_list.restype = ctypes.c_void_p
    lib.ob_pipeline_get_stream_profile_list.argtypes = [ctypes.c_void_p, ctypes.c_int, _err_t]
    lib.ob_stream_profile_list_get_profile.restype = ctypes.c_void_p
    lib.ob_stream_profile_list_get_profile.argtypes = [ctypes.c_void_p, ctypes.c_int, _err_t]
    lib.ob_delete_stream_profile_list.restype = None
    lib.ob_delete_stream_profile_list.argtypes = [ctypes.c_void_p, _err_t]

    # --- Config ---
    lib.ob_create_config.restype = ctypes.c_void_p
    lib.ob_create_config.argtypes = [_err_t]
    lib.ob_delete_config.restype = None
    lib.ob_delete_config.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_config_enable_stream.restype = None
    lib.ob_config_enable_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _err_t]

    # --- Frame ---
    lib.ob_frameset_depth_frame.restype = ctypes.c_void_p
    lib.ob_frameset_depth_frame.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_frameset_color_frame.restype = ctypes.c_void_p
    lib.ob_frameset_color_frame.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_frame_width.restype = ctypes.c_int
    lib.ob_frame_width.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_frame_height.restype = ctypes.c_int
    lib.ob_frame_height.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_frame_data.restype = ctypes.c_void_p
    lib.ob_frame_data.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_frame_data_size.restype = ctypes.c_int
    lib.ob_frame_data_size.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_depth_frame_get_value_scale.restype = ctypes.c_float
    lib.ob_depth_frame_get_value_scale.argtypes = [ctypes.c_void_p, _err_t]
    lib.ob_delete_frame.restype = None
    lib.ob_delete_frame.argtypes = [ctypes.c_void_p, _err_t]

    # --- Error handling ---
    lib.ob_error_message.restype = ctypes.c_char_p
    lib.ob_error_message.argtypes = [ctypes.c_void_p]
    lib.ob_error_function.restype = ctypes.c_char_p
    lib.ob_error_function.argtypes = [ctypes.c_void_p]
    lib.ob_delete_error.restype = None
    lib.ob_delete_error.argtypes = [ctypes.c_void_p]

    # --- Pipeline device access ---
    lib.ob_pipeline_get_device.restype = ctypes.c_void_p
    lib.ob_pipeline_get_device.argtypes = [ctypes.c_void_p, _err_t]

    # --- Device bool property ---
    lib.ob_device_set_bool_property.restype = None
    lib.ob_device_set_bool_property.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, _err_t]
    lib.ob_device_get_bool_property.restype = ctypes.c_bool
    lib.ob_device_get_bool_property.argtypes = [ctypes.c_void_p, ctypes.c_int, _err_t]

# 辅助常量
_NULL = ctypes.c_void_p(0)

# Property.h: OB_PROP_LASER_BOOL = 3
_OB_PROP_LASER_BOOL = 3


class OrbbecDepthCamera:
    """
    Orbbec Astra Pro Plus 深度相机

    架构：
    - 后台 drain 线程持续从 SDK 取帧→解析深度→缓存→释放，
      确保 SDK 内存池不会无限堆积（30fps 生产 → 30fps 消费）。
    - get_frames() 直接从缓存读取最新深度图，不阻塞。
    """

    def __init__(self):
        self._lib = None
        self._ctx = None
        self._pipeline = None
        self._config = None
        self._depth_w = 640
        self._depth_h = 480
        self._started = False

        # 后台 drain 线程
        self._grab_thread = None  # type: threading.Thread | None
        self._grab_running = False
        self._latest_depth = None  # type: np.ndarray | None
        self._lock = threading.Lock()

        # 熔断重启计数器（防止无限重启填满日志）
        self._restart_count = 0
        self._laser_enabled = False  # 是否已显式开启激光发射器

    def _check_error(self, err_ptr, context=""):
        """检查 ob_error 指针，如有错误则记录日志并释放"""
        if err_ptr and err_ptr.value and self._lib:
            _lib = self._lib
            try:
                msg = _lib.ob_error_message(err_ptr.value)
                func = _lib.ob_error_function(err_ptr.value)
                msg_str = msg.decode() if msg else "unknown"
                func_str = func.decode() if func else "?"
                logger.warning(f"[Orbbec] {context}: {msg_str} (func={func_str})")
            except Exception:
                pass
            try:
                _lib.ob_delete_error(err_ptr.value)
            except Exception:
                pass
            err_ptr.value = 0  # 重置为 NULL

    def connect(self, enable_color: bool = False) -> bool:
        """连接设备并启动流 + 后台 drain 线程"""
        try:
            self._lib = _load_lib()
            _lib = self._lib
            self._ctx = _lib.ob_create_context()

            # 显示设备信息
            dev_list = _lib.ob_query_device_list(self._ctx, ctypes.byref(_NULL))
            dev_count = _lib.ob_device_list_device_count(dev_list)
            logger.info(f"发现 {dev_count} 个 Orbbec 设备")

            if dev_count == 0:
                _lib.ob_delete_device_list(dev_list, ctypes.byref(_NULL))
                logger.warning("未发现 Orbbec 深度相机设备")
                return False

            for i in range(dev_count):
                name = _lib.ob_device_list_get_device_name(dev_list, i)
                pid = _lib.ob_device_list_get_device_pid(dev_list, i)
                sn = _lib.ob_device_list_get_device_serial_number(dev_list, i)
                logger.info(f"  [{i}] {name.decode() if name else 'N/A'}, "
                            f"PID=0x{pid:04x}, SN={sn.decode() if sn else 'N/A'}")
            _lib.ob_delete_device_list(dev_list, ctypes.byref(_NULL))

            # 创建 Pipeline
            self._pipeline = _lib.ob_create_pipeline(ctypes.byref(_NULL))
            if not self._pipeline:
                logger.error("创建 Orbbec Pipeline 失败")
                return False

            # 获取深度 profile (Astra Pro Plus 用 OB_SENSOR_IR=3)
            profiles = _lib.ob_pipeline_get_stream_profile_list(
                self._pipeline, OB_SENSOR_IR, ctypes.byref(_NULL))
            depth_vp = _lib.ob_stream_profile_list_get_profile(
                profiles, OB_PROFILE_DEFAULT, ctypes.byref(_NULL))

            self._config = _lib.ob_create_config(ctypes.byref(_NULL))
            _lib.ob_config_enable_stream(self._config, depth_vp, ctypes.byref(_NULL))
            _lib.ob_delete_stream_profile_list(profiles, ctypes.byref(_NULL))

            # 可选彩色流
            if enable_color:
                color_profiles = _lib.ob_pipeline_get_stream_profile_list(
                    self._pipeline, OB_SENSOR_COLOR, ctypes.byref(_NULL))
                color_vp = _lib.ob_stream_profile_list_get_profile(
                    color_profiles, OB_PROFILE_DEFAULT, ctypes.byref(_NULL))
                if color_vp:
                    _lib.ob_config_enable_stream(self._config, color_vp, ctypes.byref(_NULL))
                _lib.ob_delete_stream_profile_list(color_profiles, ctypes.byref(_NULL))

            # 启动 Pipeline（使用真实 error 指针，捕获 SDK 层错误）
            start_err = ctypes.c_void_p(0)
            _lib.ob_pipeline_start_with_config(
                self._pipeline, self._config, ctypes.byref(start_err))
            self._check_error(start_err, "pipeline_start_with_config")
            self._started = True
            logger.info(f"Orbbec 深度相机已启动: {self._depth_w}x{self._depth_h}")

            # ★ 尝试显式开启激光发射器（Astra Pro Plus 等 PrimeSense 方案设备）
            # 部分老旧结构光相机需要显式打开 IR 投射器才会有深度图输出
            # OB_PROP_LASER_BOOL = 3，设置为 true 打开激光
            try:
                dev = _lib.ob_pipeline_get_device(
                    self._pipeline, ctypes.byref(_NULL))
                if dev:
                    laser_err = ctypes.c_void_p(0)
                    _lib.ob_device_set_bool_property(
                        dev, _OB_PROP_LASER_BOOL, True, ctypes.byref(laser_err))
                    self._check_error(laser_err, "set laser bool")
                    # 无论设置成功与否都记录状态
                    laser_val = _lib.ob_device_get_bool_property(
                        dev, _OB_PROP_LASER_BOOL, ctypes.byref(_NULL))
                    self._laser_enabled = bool(laser_val)
                    if self._laser_enabled:
                        logger.info("Orbbec 激光发射器已开启 (OB_PROP_LASER_BOOL=true)")
                    else:
                        logger.debug("Orbbec 激光发射器状态: false（设备可能自动管理）")
            except Exception as le:
                logger.debug(f"激光发射器显式控制不支持（设备自动管理）: {le}")

            # 启动后台 drain 线程（持续取帧→缓存→释放，防止 SDK 内存池溢出）
            self._grab_running = True
            self._grab_thread = threading.Thread(
                target=self._grab_loop, daemon=True, name="OrbbecDrain")
            self._grab_thread.start()
            logger.info("Orbbec 后台帧缓存线程已启动")

            return True

        except Exception as e:
            logger.error(f"Orbbec 深度相机连接失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # --- 后台帧缓存线程 ---

    def _grab_loop(self):
        """
        后台线程：以最高速率从 SDK 取帧→释放，防止内存池堆积。
        只对最新的帧做深度解析和缓存（跳帧策略），减少 CPU 开销。

        熔断机制：
        - 连续超时到达阈值 → 自动 stop/start pipeline 恢复
        - 超时后退避 sleep，杜绝 460次/秒的忙等空转
        - 避免 pipeline 假死后把 2GB 内存吃满、整个进程 SIGABRT

        注意：Astra Pro Plus 走 OpenNI 协议 + USB2.0，SDK 内部 Pipeline 构造时
        enable_frame_sync() 会报 "not support"（SDK 已自行 catch 并继续），不影响取帧。
        """
        self._grab_loop_impl()

    # 熔断器常量（类级别，方便排查时调整）
    _FRAME_TIMEOUT_MS = 200       # timeout: 帧间隔 ~33ms，200ms 足够捕获
    _MAX_CONSECUTIVE_TIMEOUTS = 50  # 连续超时阈值（~10s 无帧即熔断）
    _MAX_RESTART_COUNT = 5         # 总重启次数上限（超标则彻底放弃）
    _TIMEOUT_BACKOFF_S = 0.02      # 超时后退避，避免 CPU 空转
    _PARSE_EVERY = 3               # 每 N 帧解析一次深度（约 10Hz）

    def _restart_pipeline(self, _lib) -> bool:
        """尝试重启 pipeline（stop → sleep → start），使用真实 error 指针"""
        self._restart_count += 1
        if self._restart_count > self._MAX_RESTART_COUNT:
            logger.critical(
                f"Orbbec pipeline 已连续重启失败 {self._restart_count - 1} 次（上限 "
                f"{self._MAX_RESTART_COUNT}），深度相机彻底不可用，"
                f"YOLO 避障将退化到无深度模式")
            self._started = False
            return False

        logger.warning(
            f"Orbbec pipeline 熔断触发，尝试重启 (第 {self._restart_count}/{self._MAX_RESTART_COUNT} 次)…")
        try:
            stop_err = ctypes.c_void_p(0)
            _lib.ob_pipeline_stop(self._pipeline, ctypes.byref(stop_err))
            self._check_error(stop_err, "pipeline_stop")
        except Exception as e:
            logger.debug(f"pipeline stop 异常（可忽略）: {e}")
        time.sleep(0.5)
        try:
            start_err = ctypes.c_void_p(0)
            _lib.ob_pipeline_start_with_config(
                self._pipeline, self._config, ctypes.byref(start_err))
            self._check_error(start_err, "pipeline_start (retry)")
            # 重启成功 → 重置计数
            self._restart_count = 0
            logger.info("Orbbec pipeline 熔断重启成功")
            return True
        except Exception as e:
            logger.error(f"Orbbec pipeline 重启失败 (#{self._restart_count}): {e}")
            return False

    def _grab_loop_impl(self):
        _lib = self._lib
        warned = False
        frame_idx = 0
        consecutive_timeouts = 0
        timeout_error_logged = False  # 只打一次超时 error 日志

        # 延迟导入，避免循环依赖
        from .config import YOLO_PAUSE_EVENT as _pause_event

        while self._grab_running and self._started:
            try:
                # ★ 使用真实 error 指针 — 超时/配置错误不会再是黑盒
                err = ctypes.c_void_p(0)
                fs = _lib.ob_pipeline_wait_for_frameset(
                    self._pipeline, self._FRAME_TIMEOUT_MS, ctypes.byref(err))

                if not fs:
                    consecutive_timeouts += 1

                    if consecutive_timeouts >= self._MAX_CONSECUTIVE_TIMEOUTS:
                        # ★ 只在熔断触发时检查一次 error，获取 SDK 真实错误原因
                        self._check_error(err, "wait_for_frameset")
                        if not timeout_error_logged:
                            timeout_error_logged = True
                            logger.error(
                                f"Orbbec 深度流连续 {consecutive_timeouts} 次超时，"
                                f"触发熔断重启")
                        if self._restart_pipeline(_lib):
                            consecutive_timeouts = 0
                            timeout_error_logged = False
                        else:
                            # 重启失败（含达到上限）→ 退出循环
                            if not self._started:
                                break
                            time.sleep(2.0)
                            consecutive_timeouts = 0
                    else:
                        time.sleep(self._TIMEOUT_BACKOFF_S)
                    continue

                # 成功取帧 → 重置所有计数
                consecutive_timeouts = 0
                timeout_error_logged = False
                frame_idx += 1

                # 每 300 帧（约 10 秒）记录一次 drain 速率
                if frame_idx % 300 == 0:
                    logger.debug(f"Orbbec drain 线程: 已处理 {frame_idx} 帧")

                # 非障碍检测模式（VLM 推理期间）：全速 drain + 释放，不解析
                if not _pause_event.is_set():
                    _lib.ob_delete_frame(fs, ctypes.byref(_NULL))
                    continue

                # 障碍检测模式：每隔 N 帧解析深度（降低 CPU）
                if frame_idx % self._PARSE_EVERY == 0:
                    df = _lib.ob_frameset_depth_frame(fs, ctypes.byref(_NULL))
                    if df:
                        try:
                            w = _lib.ob_frame_width(df, ctypes.byref(_NULL))
                            h = _lib.ob_frame_height(df, ctypes.byref(_NULL))
                            sz = _lib.ob_frame_data_size(df, ctypes.byref(_NULL))
                            scale = _lib.ob_depth_frame_get_value_scale(df, ctypes.byref(_NULL))
                            ptr = _lib.ob_frame_data(df, ctypes.byref(_NULL))

                            if ptr and sz > 0 and w > 0 and h > 0:
                                buf = (ctypes.c_uint16 * (w * h)).from_address(ptr)
                                depth_map = np.ctypeslib.as_array(buf).reshape(h, w).astype(np.float32).copy()
                                depth_map *= scale

                                with self._lock:
                                    self._latest_depth = depth_map

                                if not warned:
                                    warned = True
                        finally:
                            # ★ 必须释放 depth frame 引用，防止 SDK 引用计数泄漏
                            _lib.ob_delete_frame(df, ctypes.byref(_NULL))

                # 释放 frameset
                _lib.ob_delete_frame(fs, ctypes.byref(_NULL))

            except Exception as e:
                logger.error(f"Orbbec 后台线程异常: {e}")
                consecutive_timeouts += 1
                time.sleep(0.1)

        logger.info(f"Orbbec 后台帧缓存线程已退出 (共处理 {frame_idx} 帧, "
                    f"started={self._started}, restart_count={self._restart_count})")

    # --- 对外接口 ---

    def get_frames(self, timeout_ms: int = 100):
        """
        从缓存获取最新深度图（非阻塞）。

        Args:
            timeout_ms: 兼容参数，当前版本忽略（从缓存读取）

        Returns:
            (depth_map_mm, None)  - 只返回深度图
        """
        with self._lock:
            if self._latest_depth is not None:
                return self._latest_depth.copy(), None
        return None, None

    # --- Web 预览 ---

    def get_latest_depth_colormap(self, max_depth_mm: int = 6000) -> Optional[np.ndarray]:
        """
        返回最新深度图的伪彩色图像（JET 色系），供 web 预览等场景使用。

        Args:
            max_depth_mm: 深度可视化的最大值 (mm)，超过此值截断

        Returns:
            BGR 格式的 numpy 数组，无有效深度时返回 None
        """
        with self._lock:
            if self._latest_depth is None:
                return None
            depth_mm = self._latest_depth.copy()

        invalid = (depth_mm <= 0)
        clipped = np.clip(depth_mm, 0, max_depth_mm)
        normalized = (clipped / max_depth_mm * 255).astype(np.uint8)

        try:
            import cv2
            colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_JET)
            colored[invalid] = (0, 0, 0)
            return colored
        except ImportError:
            logger.warning("cv2 不可用，无法生成深度伪彩色图")
            return None

    # --- 便捷方法 ---

    def get_center_depth(self, depth_map) -> float:
        """获取画面中心深度 (mm), 无效返回 -1"""
        if depth_map is None:
            return -1
        h, w = depth_map.shape
        return self.get_depth_at_point(depth_map, w // 2, h // 2)

    def get_depth_at_point(self, depth_map, x: int, y: int) -> float:
        """获取指定像素深度 (mm)，无效返回 -1"""
        if depth_map is None:
            return -1
        h, w = depth_map.shape
        if 0 <= x < w and 0 <= y < h:
            v = depth_map[y, x]
            return float(v) if v > 0 else -1
        return -1

    def get_bbox_depth(self, depth_map, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        获取 bbox 区域中位深度 (mm)

        Returns:
            深度值 (mm)，无效返回 -1
        """
        if depth_map is None:
            return -1
        h, w = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return -1
        roi = depth_map[y1:y2, x1:x2]
        valid = roi[roi > 0]
        return float(np.median(valid)) if len(valid) > 0 else -1

    def is_running(self) -> bool:
        return self._started

    def stop(self):
        """停止并释放资源"""
        # 先停止后台线程
        self._grab_running = False
        if self._grab_thread and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=2.0)

        if self._pipeline and self._lib:
            self._lib.ob_pipeline_stop(self._pipeline, ctypes.byref(_NULL))
            self._lib.ob_delete_pipeline(self._pipeline)
            self._pipeline = None
        if self._config and self._lib:
            self._lib.ob_delete_config(self._config, ctypes.byref(_NULL))
            self._config = None
        if self._ctx and self._lib:
            self._lib.ob_delete_context(self._ctx)
            self._ctx = None
        self._started = False
        self._lib = None

        with self._lock:
            self._latest_depth = None

        logger.info("Orbbec 深度相机已释放")

    def __del__(self):
        self.stop()

    @property
    def depth_size(self):
        return (self._depth_w, self._depth_h)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cam = OrbbecDepthCamera()
    if not cam.connect():
        exit(1)

    print(f"\n深度图: {cam.depth_size}")
    print("读取中 (Ctrl+C 退出)...\n")

    try:
        for i in range(30):
            depth, color = cam.get_frames(1000)
            if depth is not None:
                c = cam.get_center_depth(depth)
                valid = depth[depth > 0]
                if len(valid) > 0:
                    print(f"[{i}] 中心={c:.0f}mm "
                          f"范围=[{valid.min():.0f}, {valid.max():.0f}]mm "
                          f"有效={len(valid)}")
                else:
                    print(f"[{i}] 无有效深度")
            else:
                print(f"[{i}] 等待...")
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
