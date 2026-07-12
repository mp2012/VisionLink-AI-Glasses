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

# 辅助常量
_NULL = ctypes.c_void_p(0)


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

            # 启动 Pipeline
            _lib.ob_pipeline_start_with_config(self._pipeline, self._config, ctypes.byref(_NULL))
            self._started = True
            logger.info(f"Orbbec 深度相机已启动: {self._depth_w}x{self._depth_h}")

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

        策略：timeout=10ms 短阻塞 drain SDK 帧队列，每隔 N 帧才解析一次深度数据。
        确保消费速度 >= SDK 30fps 生产速度，防止 2GB 内存池溢出。
        """
        self._grab_loop_impl()

    def _grab_loop_impl(self):
        _lib = self._lib
        depth_w, depth_h = self._depth_w, self._depth_h
        warned = False
        parse_every = 3  # 每 3 帧解析一次深度（约 10Hz，足够 YOLO 使用）
        frame_idx = 0

        # 延迟导入，避免循环依赖
        from .config import YOLO_PAUSE_EVENT as _pause_event

        while self._grab_running and self._started:
            # 非障碍检测模式：暂停取帧，释放 CPU 和 SDK 内存资源
            if not _pause_event.is_set():
                _pause_event.wait(timeout=0.5)
                continue

            try:
                # timeout=100ms：SDK 30fps 帧间隔约 33ms，100ms 足以在正常取帧的同时
                # 也留出缓冲空间。同时通过 OrbbecSDKConfig.xml 将 ConsoleLogLevel=5(OFF)
                # 来彻底消除 "Wait for frame timeout" 日志噪音。
                fs = _lib.ob_pipeline_wait_for_frameset(
                    self._pipeline, 100, ctypes.byref(_NULL))
                if not fs:
                    continue

                frame_idx += 1

                # 每隔 parse_every 帧才解析深度（降低 CPU 开销）
                if frame_idx % parse_every == 0:
                    df = _lib.ob_frameset_depth_frame(fs, ctypes.byref(_NULL))
                    if df:
                        w = _lib.ob_frame_width(df, ctypes.byref(_NULL))
                        h = _lib.ob_frame_height(df, ctypes.byref(_NULL))
                        sz = _lib.ob_frame_data_size(df, ctypes.byref(_NULL))
                        scale = _lib.ob_depth_frame_get_value_scale(df, ctypes.byref(_NULL))
                        ptr = _lib.ob_frame_data(df, ctypes.byref(_NULL))

                        if ptr and sz > 0 and w > 0 and h > 0:
                            buf = (ctypes.c_uint16 * (w * h)).from_address(ptr)
                            depth_map = np.ctypeslib.as_array(buf).reshape(h, w).astype(np.float32).copy()
                            depth_map *= scale
                            depth_w, depth_h = w, h

                            with self._lock:
                                self._latest_depth = depth_map

                            if not warned:
                                warned = True

                # 释放 frameset（关键：SDK 内存池回收）
                _lib.ob_delete_frame(fs, ctypes.byref(_NULL))

            except Exception as e:
                logger.error(f"Orbbec 后台线程异常: {e}")
                time.sleep(0.1)

        logger.debug("Orbbec 后台帧缓存线程已退出")

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
