"""
摄像头管理模块 - 双摄像头协同架构
支持 POV（镜腿单目）和 FOV（胸前深度相机）双路独立管理

架构说明：
- POV Camera: 随头部转动，高清晰度，按需截帧送大模型推理
- FOV Camera: 固定胸前，常开流式，送 YOLO 实时避障检测
"""
import time
import logging
import threading
from typing import Optional, Tuple, Any

import cv2

from .platform import IS_JETSON, IS_WINDOWS
from .config import POV_CAMERA_CONFIG, FOV_CAMERA_CONFIG

logger = logging.getLogger(__name__)


class CameraManager:
    """
    通用单摄像头管理器
    封装跨平台摄像头操作：打开、读取、释放
    """

    def __init__(self, config: dict = None, name: str = "Camera"):
        """
        Args:
            config: 摄像头配置字典，默认使用 POV_CAMERA_CONFIG
            name: 摄像头名称标识（用于日志）
        """
        self.config = config or POV_CAMERA_CONFIG
        self.name = name
        self.cap: Optional[cv2.VideoCapture] = None
        self.width = self.config.get("width", 640)
        self.height = self.config.get("height", 480)
        self._lock = threading.Lock()
        self._reconnect_attempts = 0
        self._max_reconnect = 5

    def open(self) -> bool:
        """打开摄像头，自动适配平台后端"""
        cam_id = self.config.get("cam_id", 0)

        if IS_JETSON:
            return self._open_jetson()
        elif IS_WINDOWS:
            return self._open_windows(cam_id)
        else:
            return self._open_default(cam_id)

    def _open_jetson(self) -> bool:
        """Jetson: 优先 GStreamer MJPG pipeline，回退到索引扫描"""
        width, height = self.width, self.height
        use_gst = self.config.get("gstreamer_pipeline", True)

        if use_gst:
            # GStreamer MJPG pipeline（兼容 Astra Plus 等 USB 摄像头）
            cam_id = self.config.get("cam_id", 0)
            pipe = (
                f"v4l2src device=/dev/video{cam_id} ! "
                f"image/jpeg,width={width},height={height},framerate=30/1 ! "
                f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
            )
            try:
                cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        logger.info(
                            f"[{self.name}] Jetson 摄像头已锁定: /dev/video{cam_id} "
                            f"(GStreamer MJPG, {width}x{height})"
                        )
                        self._reconnect_attempts = 0
                        return True
                    cap.release()
            except Exception as e:
                logger.warning(f"[{self.name}] GStreamer 管道失败: {e}")

        # 回退：遍历摄像头 ID
        for cam_id in self.config.get("auto_scan_ids", [0, 1, 2, 4, 5]):
            for backend in [None, cv2.CAP_V4L2]:
                try:
                    cap = cv2.VideoCapture(cam_id) if backend is None else cv2.VideoCapture(cam_id, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.cap = cap
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                            backend_name = "default" if backend is None else "V4L2"
                            logger.info(
                                f"[{self.name}] Jetson 摄像头已锁定 ID: {cam_id} "
                                f"(backend={backend_name})"
                            )
                            self._reconnect_attempts = 0
                            return True
                        cap.release()
                except Exception as e:
                    logger.warning(f"[{self.name}] 摄像头 ID {cam_id} 打开失败: {e}")

        logger.error(f"[{self.name}] Jetson 未检测到可用摄像头")
        return False

    def _open_windows(self, cam_id: int) -> bool:
        """Windows: DSHOW 优先，自动降级"""
        try:
            self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                logger.warning(f"[{self.name}] CAP_DSHOW 打开失败，尝试默认模式")
                self.cap = cv2.VideoCapture(cam_id)
            if not self.cap.isOpened():
                logger.error(f"[{self.name}] Windows 摄像头 {cam_id} 打开失败")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            logger.info(f"[{self.name}] Windows 摄像头初始化完成，{self.width}x{self.height}")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Windows 摄像头异常: {e}")
            return False

    def _open_default(self, cam_id: int) -> bool:
        """通用默认打开方式"""
        try:
            self.cap = cv2.VideoCapture(cam_id)
            if not self.cap.isOpened():
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True
        except Exception as e:
            logger.error(f"[{self.name}] 摄像头打开失败: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[Any]]:
        """
        读取一帧（线程安全）

        Returns:
            (success, frame): 成功标志和帧数据
        """
        with self._lock:
            if self.cap is None:
                return False, None
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning(f"[{self.name}] 读取帧失败，尝试重连...")
                    self._try_reconnect()
                return ret, frame
            except Exception as e:
                logger.error(f"[{self.name}] 读取异常: {e}")
                return False, None

    def _try_reconnect(self) -> bool:
        """自动重连机制"""
        if self._reconnect_attempts >= self._max_reconnect:
            logger.error(f"[{self.name}] 已达最大重连次数 ({self._max_reconnect})，放弃")
            return False

        self._reconnect_attempts += 1
        logger.info(
            f"[{self.name}] 第 {self._reconnect_attempts}/{self._max_reconnect} 次重连尝试..."
        )
        self.release()
        time.sleep(0.5)
        return self.open()

    def is_opened(self) -> bool:
        """检查摄像头是否已打开"""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """释放摄像头资源"""
        with self._lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.warning(f"[{self.name}] 释放摄像头异常: {e}")
                self.cap = None

    def get_resolution(self) -> Tuple[int, int]:
        """获取当前分辨率"""
        return self.width, self.height

    @staticmethod
    def crop_roi(frame, h_ratio=(0.25, 0.75), v_ratio=(0.2, 0.8)):
        """裁剪画面中央 ROI 区域"""
        h, w = frame.shape[:2]
        x1, x2 = int(w * h_ratio[0]), int(w * h_ratio[1])
        y1, y2 = int(h * v_ratio[0]), int(h * v_ratio[1])
        return frame[y1:y2, x1:x2]


class DualCameraManager:
    """
    双摄像头管理器
    同时管理 POV（镜腿单目）和 FOV（胸前深度）两路摄像头

    Usage:
        manager = DualCameraManager()
        if manager.open_both():
            while True:
                pov_ret, pov_frame = manager.read_pov()
                fov_ret, fov_frame = manager.read_fov()
    """

    def __init__(self):
        self.pov = CameraManager(POV_CAMERA_CONFIG, name="POV-镜腿单目")
        self.fov = CameraManager(FOV_CAMERA_CONFIG, name="FOV-胸前深度")
        self._fov_enabled = False
        self._pov_enabled = False

    def open_pov(self) -> bool:
        """仅打开 POV 摄像头"""
        self._pov_enabled = self.pov.open()
        return self._pov_enabled

    def open_fov(self) -> bool:
        """仅打开 FOV 摄像头"""
        self._fov_enabled = self.fov.open()
        return self._fov_enabled

    def open_both(self) -> Tuple[bool, bool]:
        """
        打开双路摄像头

        Returns:
            (pov_ok, fov_ok): 两路是否成功
        """
        pov_ok = self.open_pov()
        fov_ok = self.open_fov()
        if not pov_ok:
            logger.warning("POV 摄像头未就绪，大模型推理功能受限")
        if not fov_ok:
            logger.warning("FOV 摄像头未就绪，YOLO 避障功能将禁用")
        return pov_ok, fov_ok

    def read_pov(self) -> Tuple[bool, Optional[Any]]:
        """读取 POV 摄像头帧"""
        if not self._pov_enabled:
            return False, None
        return self.pov.read()

    def read_fov(self) -> Tuple[bool, Optional[Any]]:
        """读取 FOV 摄像头帧"""
        if not self._fov_enabled:
            return False, None
        return self.fov.read()

    def is_pov_ok(self) -> bool:
        return self._pov_enabled and self.pov.is_opened()

    def is_fov_ok(self) -> bool:
        return self._fov_enabled and self.fov.is_opened()

    def release_all(self):
        """释放全部摄像头"""
        self.pov.release()
        self.fov.release()
        self._pov_enabled = False
        self._fov_enabled = False
