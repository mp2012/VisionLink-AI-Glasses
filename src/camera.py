"""
摄像头管理模块
自动适配 Windows (DSHOW) 和 Jetson (V4L2)，支持自动遍历和 ROI 裁剪。
"""
import time
import logging
import cv2

from .platform import IS_JETSON, IS_WINDOWS
from .config import CAMERA_CONFIG

logger = logging.getLogger(__name__)


class CameraManager:
    """跨平台摄像头管理器"""

    def __init__(self):
        self.cap = None
        self.width = CAMERA_CONFIG["width"]
        self.height = CAMERA_CONFIG["height"]

    def open(self) -> bool:
        """打开摄像头，自动适配平台后端"""
        cam_id = CAMERA_CONFIG["cam_id"]
        backend = CAMERA_CONFIG["backend"]

        if IS_JETSON:
            return self._open_jetson()
        elif IS_WINDOWS:
            return self._open_windows(cam_id)
        else:
            return self._open_default(cam_id)

    def _open_jetson(self) -> bool:
        """Jetson: 遍历摄像头 ID，使用 V4L2 后端"""
        for cam_id in CAMERA_CONFIG.get("auto_scan_ids", [0, 1, 2, 4, 5]):
            cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    logger.info(f"Jetson 摄像头已锁定 ID: {cam_id}")
                    return True
                cap.release()
        logger.error("Jetson 未检测到可用摄像头")
        return False

    def _open_windows(self, cam_id: int) -> bool:
        """Windows: DSHOW 优先，自动降级"""
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.warning("CAP_DSHOW 打开失败，尝试默认模式")
            self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            logger.error(f"Windows 摄像头 {cam_id} 打开失败")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info(f"Windows 摄像头初始化完成，分辨率：{self.width}×{self.height}")
        return True

    def _open_default(self, cam_id: int) -> bool:
        """通用默认打开方式"""
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def read(self):
        """读取一帧"""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    @staticmethod
    def crop_roi(frame, h_ratio=(0.25, 0.75), v_ratio=(0.2, 0.8)):
        """裁剪画面中央 ROI 区域"""
        h, w = frame.shape[:2]
        x1, x2 = int(w * h_ratio[0]), int(w * h_ratio[1])
        y1, y2 = int(h * v_ratio[0]), int(h * v_ratio[1])
        return frame[y1:y2, x1:x2]
