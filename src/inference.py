"""
Ollama 多模态推理模块
封装图像预处理、Base64 编码、模型调用、超时控制。
"""
import time
import base64
import logging
import threading

import cv2
import ollama

from .platform import IS_JETSON
from .config import MODEL_NAME, AI_IMAGE_SIZE, JPEG_QUALITY, TIMEOUT_INFER, INFER_OPTIONS, STATE_IDLE, STATE_INFER

logger = logging.getLogger(__name__)


class InferenceEngine:
    """跨平台多模态推理引擎"""

    def __init__(self):
        self._state = STATE_IDLE
        self._lock = threading.Lock()

    @property
    def is_busy(self) -> bool:
        return self._state != STATE_IDLE

    @staticmethod
    def image_to_base64(frame):
        """将 OpenCV 帧缩放后编码为 Base64"""
        h, w = frame.shape[:2]
        scale = AI_IMAGE_SIZE / max(w, h) if max(w, h) > AI_IMAGE_SIZE else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        resize_img = cv2.resize(frame, (new_w, new_h))
        success, buf = cv2.imencode(".jpg", resize_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not success:
            logger.error("图像编码失败")
            return None
        return base64.b64encode(buf).decode("utf-8")

    def infer(self, prompt: str, img_b64: str = None) -> str:
        """同步推理（带状态锁）"""
        with self._lock:
            if self._state != STATE_IDLE:
                logger.warning("系统忙，拒绝本次推理请求")
                return ""
            self._state = STATE_INFER

        try:
            logger.info("开始调用大模型推理")
            messages = [{"role": "user", "content": prompt}]
            if img_b64:
                messages[0]["images"] = [img_b64]

            resp = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                options=INFER_OPTIONS,
            )
            result = resp["message"]["content"].strip()
            logger.info(f"推理完成(长度={len(result)}): {repr(result[:120])}")
            return result
        except Exception as e:
            logger.error(f"推理异常：{e}")
            return ""
        finally:
            self._state = STATE_IDLE

    def infer_async(self, prompt: str, img_b64: str, callback, *args):
        """异步推理，结果通过 callback(result, *args) 返回"""

        def _worker():
            result = self.infer(prompt, img_b64)
            if callback:
                callback(result, *args)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        # Jetson 环境需要超时兜底
        if IS_JETSON:
            t.join(timeout=TIMEOUT_INFER)
            if t.is_alive():
                logger.warning(f"推理超时 {TIMEOUT_INFER}s，强制释放")
                with self._lock:
                    self._state = STATE_IDLE
        return t
