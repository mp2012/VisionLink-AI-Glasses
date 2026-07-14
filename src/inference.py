"""
Ollama 多模态推理模块
封装图像预处理、Base64 编码、模型调用、超时控制

特性：
- 同步 + 异步推理接口
- 图像自动缩放与 JPEG 压缩
- 状态锁防并发
- Jetson 超时兜底机制
"""
import time
import re
import base64
import logging
import threading
from typing import Optional, Callable

import cv2
import ollama

from .platform import IS_JETSON
from .config import MODEL_NAME, AI_IMAGE_SIZE, JPEG_QUALITY, TIMEOUT_INFER, INFER_OPTIONS
from .config import STATE_IDLE, STATE_INFER

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    跨平台多模态推理引擎
    封装 Ollama API 调用，支持图像+文本多模态输入

    Usage:
        engine = InferenceEngine()
        img_b64 = engine.image_to_base64(frame)
        result = engine.infer("描述这张图片", img_b64)
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_NAME
        self._state = STATE_IDLE
        self._lock = threading.Lock()

    @property
    def is_busy(self) -> bool:
        """是否正在推理中"""
        return self._state != STATE_IDLE

    @property
    def model(self) -> str:
        """当前使用的模型名称"""
        return self.model_name

    @staticmethod
    def image_to_base64(frame) -> Optional[str]:
        """
        将 OpenCV 帧缩放后编码为 Base64

        Args:
            frame: OpenCV BGR 图像帧

        Returns:
            Base64 编码字符串，失败返回 None
        """
        try:
            h, w = frame.shape[:2]
            scale = AI_IMAGE_SIZE / max(w, h) if max(w, h) > AI_IMAGE_SIZE else 1.0
            new_w, new_h = int(w * scale), int(h * scale)
            resize_img = cv2.resize(frame, (new_w, new_h))
            success, buf = cv2.imencode(
                ".jpg", resize_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not success:
                logger.error("图像编码失败")
                return None
            return base64.b64encode(buf).decode("utf-8")
        except Exception as e:
            logger.error(f"图像编码异常: {e}")
            return None

    @staticmethod
    def image_to_base64_from_path(image_path: str) -> Optional[str]:
        """
        从文件路径加载图像并编码为 Base64

        Args:
            image_path: 图像文件路径

        Returns:
            Base64 编码字符串，失败返回 None
        """
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"无法读取图像文件: {image_path}")
                return None
            return InferenceEngine.image_to_base64(frame)
        except Exception as e:
            logger.error(f"图像加载异常: {e}")
            return None

    def infer(self, prompt: str, img_b64: str = None, images: list = None) -> str:
        """
        同步推理（带状态锁，防止并发调用）

        Args:
            prompt: 提示词文本
            img_b64: 单张图片的 Base64 编码
            images: 多张图片的 Base64 编码列表

        Returns:
            模型推理结果文本，失败返回空字符串
        """
        with self._lock:
            if self._state != STATE_IDLE:
                logger.warning("系统忙，拒绝本次推理请求")
                return ""
            self._state = STATE_INFER

        t_start = time.time()
        try:
            logger.info(f"开始调用模型推理 [{self.model_name}]")

            # 构建消息
            messages = [{"role": "user", "content": prompt}]

            # 处理图像
            img_list = images or []
            if img_b64:
                img_list = [img_b64] + img_list
            if img_list:
                messages[0]["images"] = img_list

            resp = ollama.chat(
                model=self.model_name,
                messages=messages,
                options=INFER_OPTIONS,
                keep_alive=-1,  # -1 = 模型常驻内存，避免每次冷加载（Jetson 30-50s 延迟的根因）
            )
            latency_ms = (time.time() - t_start) * 1000
            msg = resp.get("message", {})
            result = msg.get("content", "").strip()

            # 兼容 thinking 模型（gemma4 等）：content 为空时从 thinking 中提取最终答案
            if not result:
                thinking = msg.get("thinking", "").strip()
                if thinking:
                    extracted = self._extract_final_answer(thinking)
                    if extracted:
                        logger.info(f"推理完成 (thinking提取, 长度={len(extracted)}): {repr(extracted[:120])}")
                        self._log_inference_dashboard(extracted, True, latency_ms)
                        return extracted
                    else:
                        # thinking 未正常结束，丢弃草稿，不播英文内部推理流程
                        logger.warning(
                            f"thinking 未正常结束，丢弃草稿 (thinking长度={len(thinking)}): "
                            f"{repr(thinking[:80])}"
                        )
                        self._log_inference_dashboard("", False, latency_ms)
                        return ""

            logger.info(f"推理完成 (长度={len(result)}): {repr(result[:120])}")
            self._log_inference_dashboard(result, True, latency_ms)
            return result

        except ollama.ResponseError as e:
            logger.error(f"Ollama 响应错误: {e}")
            self._log_inference_dashboard("", False, 0)
            return ""
        except Exception as e:
            logger.error(f"推理异常: {e}")
            self._log_inference_dashboard("", False, 0)
            return ""
        finally:
            self._state = STATE_IDLE

    @staticmethod
    def _log_inference_dashboard(text: str, success: bool, latency_ms: float):
        """仪表板打点：记录推理结果与 Ollama 连接状态"""
        try:
            from src.dashboard_status import system_status
            if success:
                system_status.set_ollama_connected(True)
            else:
                system_status.set_ollama_connected(False)
            system_status.log_inference(text, success, latency_ms)
        except ImportError:
            pass

    @staticmethod
    def _extract_final_answer(thinking_text: str) -> str:
        """
        从 thinking 模型的推理过程中提取最终答案

        gemma4 实际输出格式多样：
            ...推理过程...
            ...done thinking.
            实际答案

            ...推理过程...
            *(Self-Correction ...)*  The answer should be...
            实际答案（中文）

        策略：
        1. 找 "...done thinking." 标记后的内容
        2. 剔除 *(Self-Correction ...)* 等元注释块
        3. 优先返回含中文的末段（最可能是用户能懂的答案）
        4. 最终回退：取最后 300 字符
        """
        text = thinking_text

        # 策略1：找 "...done thinking." 标记后的内容
        for marker in ["...done thinking.", "Done thinking.", "done thinking"]:
            idx = text.lower().find(marker.lower())
            if idx != -1:
                after = text[idx + len(marker):].strip()
                if after:
                    cleaned = InferenceEngine._clean_meta_commentary(after)
                    logger.debug(f"从 thinking 标记后提取: {repr(cleaned[:80])}")
                    return cleaned

        # 策略2：剔除 *(Self-Correction...)* 之类的元注释块
        text = InferenceEngine._clean_meta_commentary(text)

        # 策略3：按双换行分段，优先取含中文的最后一段
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return ""

        # 先找包含中文的最后一段（这最可能是用户要的答案）
        cn_paragraphs = [p for p in paragraphs if re.search(r'[\u4e00-\u9fff]', p)]
        if cn_paragraphs:
            return cn_paragraphs[-1]

        # 无中文段落时：跳过推理步骤，取最后一段
        skip_patterns = (
            "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.",
            "- ", "* ", "**",
            "analyze", "evaluate", "identify",
            "thinking process", "step ", "option ", "approach",
            "therefore", "thus", "so the", "this means",
            "key observation", "key insight", "final output",
            "the resulting", "the image shows", "the scene",
        )
        final_paragraphs = []
        for p in paragraphs:
            first_line = p.split("\n")[0].strip().lower()
            if any(first_line.startswith(sp) for sp in skip_patterns):
                continue
            final_paragraphs.append(p)

        if final_paragraphs:
            result = final_paragraphs[-1]
            # 检查是否仍然是推理步骤文本（开头包含 thinking/step/approach 等关键词）
            if any(w in result.lower()[:60] for w in ("thinking", "process", "step", "approach", "drafting", "option", "internal")):
                return ""
            return result

        # 所有策略均失败 → thinking 未正常结束，不返回草稿
        return ""

    @staticmethod
    def _clean_meta_commentary(text: str) -> str:
        """
        剔除 thinking 模型输出的元注释，如：
        *(Self-Correction during translation: ...)* The resulting descriptions should...
        """
        # 移除 *( ... )* 样式的内省注释
        text = re.sub(r'\*\([^)]*\)\*', '', text)
        # 移除行首的 *( ... )* 直到下一个句号结束的元句子
        text = re.sub(r'^\*\([^)]*\)\*\s*[A-Z][^。.!！?\n]{20,200}[.!]\s*', '', text, flags=re.MULTILINE)
        return text.strip()

    def infer_async(
        self,
        prompt: str,
        img_b64: str = None,
        callback: Callable = None,
        *callback_args
    ):
        """
        异步推理，结果通过 callback(result, *args) 返回

        Args:
            prompt: 提示词文本
            img_b64: 图片 Base64 编码
            callback: 回调函数
            *callback_args: 回调函数额外参数

        Returns:
            threading.Thread 对象
        """
        def _worker():
            result = self.infer(prompt, img_b64)
            if callback:
                try:
                    callback(result, *callback_args)
                except Exception as e:
                    logger.error(f"推理回调异常: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Jetson 环境需要超时兜底（仅告警，不重置状态）
        # 状态由 infer() 的 finally 块在线程结束后自然重置
        if IS_JETSON:
            t.join(timeout=TIMEOUT_INFER)
            if t.is_alive():
                logger.warning(f"推理超时 {TIMEOUT_INFER}s，等待后台线程完成")

        return t
