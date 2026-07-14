"""
UI 绘制模块
用于开发/演示端通过 OpenCV/PIL 渲染带有 YOLO 检测框、状态面板的合成画面

特性：
- 半透明侧面板（显示模式、状态、操作指引）
- YOLO 检测框叠加（带颜色分级）
- 无头模式自动适配（Null Object Pattern）
- 中文字体渲染（PIL）
"""
import os
import logging
from typing import Optional, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .platform import IS_JETSON, IS_WINDOWS, HAS_DISPLAY
from .config import UI_PANEL_WIDTH, UI_ALPHA, UI_YOLO_BOX_COLOR, UI_DANGER_COLOR, UI_WARNING_COLOR
from .config import FONT_PATHS, FONT_SIZES, STATE_IDLE, STATE_INFER, STATE_TTS
from .prompts import STATE_NAME_LIST, MODE_NAME_LIST

logger = logging.getLogger(__name__)


class UIManager:
    """
    UI 管理器
    在开发/演示端渲染合成画面，无头模式下自动跳过绘制

    Usage:
        ui = UIManager()
        ui.enable_gui = True  # 或 False（无头模式）
        output = ui.render(frame, state, mode, info_texts)
    """

    def __init__(self, enable_gui: bool = None):
        """
        Args:
            enable_gui: 是否启用 GUI 渲染。None 表示自动检测。
        """
        if enable_gui is None:
            enable_gui = HAS_DISPLAY and not IS_JETSON
        self.enable_gui = enable_gui
        self._font = None
        self._font_path = None

        if self.enable_gui:
            self._init_font()

    def _init_font(self):
        """加载中文字体"""
        for fp in FONT_PATHS:
            if os.path.exists(fp):
                self._font_path = fp
                try:
                    self._font = ImageFont.truetype(fp, FONT_SIZES[0])
                    logger.info(f"UI 字体加载: {fp}")
                    return
                except Exception:
                    pass
        logger.warning("未找到中文字体，UI 中文可能显示异常")
        self._font = ImageFont.load_default()

    # ==================== 主渲染接口 ====================

    def render(
        self,
        frame,
        state: int = STATE_IDLE,
        current_mode: int = 1,
        info_texts: list = None,
        yolo_result=None,
        auto_enabled: bool = False,
        lang: str = "zh",
    ) -> np.ndarray:
        """
        渲染合成画面

        Args:
            frame: 原始帧
            state: 当前状态
            current_mode: 当前模式编号
            info_texts: 额外信息文本列表
            yolo_result: YOLO 检测结果 (DetectionResult)
            auto_enabled: 自动模式是否开启
            lang: 语种

        Returns:
            合成后的帧（无头模式直接返回原帧）
        """
        if not self.enable_gui:
            return frame

        output = frame.copy()

        # 叠加 YOLO 检测框
        if yolo_result is not None and hasattr(yolo_result, 'objects'):
            output = self._draw_yolo_boxes(output, yolo_result)

        # 叠加状态面板
        output = self._draw_panel(output, state, current_mode, info_texts, auto_enabled, lang)

        return output

    def render_fov_debug(self, fov_frame, yolo_detector):
        """
        渲染 FOV 调试画面（YOLO 检测框 + 统计信息）

        Args:
            fov_frame: FOV 摄像头原始帧
            yolo_detector: YOLODetector 实例

        Returns:
            标注后的帧
        """
        if not self.enable_gui or yolo_detector is None:
            return fov_frame

        annotated = yolo_detector.annotate_frame(fov_frame)

        # 叠加检测统计
        stats = yolo_detector.get_stats()
        stats_text = f"FOV YOLO | Frames: {stats['frame_count']} | Detects: {stats['detect_count']}"
        cv2.putText(
            annotated, stats_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        return annotated

    # ==================== 内部绘制方法 ====================

    def _draw_yolo_boxes(self, frame, yolo_result) -> np.ndarray:
        """在帧上绘制 YOLO 检测框"""
        from .detection import OBSTACLE_LEVELS

        output = frame.copy()
        for obj in yolo_result.objects:
            x1, y1, x2, y2 = obj["bbox"]
            class_name = obj.get("class_name", "")
            conf = obj.get("confidence", 0)

            # 颜色分级
            if obj["class_id"] in OBSTACLE_LEVELS.get("danger", {}):
                color = UI_DANGER_COLOR
            elif obj["class_id"] in OBSTACLE_LEVELS.get("warning", {}):
                color = UI_WARNING_COLOR
            else:
                color = UI_YOLO_BOX_COLOR

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                output, label, (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        return output

    def _draw_panel(
        self,
        frame,
        state: int,
        mode: int,
        info_texts: list,
        auto_enabled: bool,
        lang: str,
    ) -> np.ndarray:
        """绘制半透明侧面板"""
        h, w = frame.shape[:2]
        panel = np.zeros((h, UI_PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        # 半透明叠加
        roi = frame[:, w - UI_PANEL_WIDTH:w]
        blended = cv2.addWeighted(roi, 1 - UI_ALPHA, panel, UI_ALPHA, 0)
        output = frame.copy()
        output[:, w - UI_PANEL_WIDTH:w] = blended

        # 使用 PIL 绘制中文
        pil_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        x_base = w - UI_PANEL_WIDTH + 10
        y = 10

        # 标题
        draw.text((x_base, y), "VisionLink", fill=(0, 255, 255), font=self._font)
        y += 35

        # 状态指示灯
        state_colors = {
            STATE_IDLE: (100, 100, 100),
            STATE_INFER: (0, 255, 255),
            STATE_TTS: (0, 255, 0),
        }
        state_names = {STATE_IDLE: STATE_NAME_LIST[lang][STATE_IDLE], STATE_INFER: STATE_NAME_LIST[lang][STATE_INFER], STATE_TTS: STATE_NAME_LIST[lang][STATE_TTS]}
        color = state_colors.get(state, (100, 100, 100))
        state_fallback = "未知" if lang == "zh" else "Unknown"
        state_text = state_names.get(state, state_fallback)
        draw.text((x_base, y), f"● {state_text}", fill=color, font=self._font)
        y += 25

        # 当前模式
        unknown_label = "未知" if lang == "zh" else "Unknown"
        mode_name = MODE_NAME_LIST[lang][mode - 1] if 1 <= mode <= 5 else unknown_label
        mode_label = "模式:" if lang == "zh" else "Mode:"
        draw.text((x_base, y), f"{mode_label} {mode_name}", fill=(255, 255, 255), font=self._font)
        y += 25

        # 自动模式
        auto_labels = {"zh": {"on": "自动: ON", "off": "自动: OFF"}, "en": {"on": "Auto: ON", "off": "Auto: OFF"}}
        auto_text = auto_labels[lang]["on"] if auto_enabled else auto_labels[lang]["off"]
        auto_color = (0, 255, 0) if auto_enabled else (150, 150, 150)
        draw.text((x_base, y), auto_text, fill=auto_color, font=self._font)
        y += 25

        # 语种
        lang_labels = {"zh": "语言: 中文", "en": "Language: EN"}
        draw.text((x_base, y), lang_labels[lang], fill=(200, 200, 200), font=self._font)
        y += 30

        # 分隔线
        draw.line([(x_base, y), (w - 10, y)], fill=(100, 100, 100), width=1)
        y += 10

        # 额外信息
        if info_texts:
            for text in info_texts:
                draw.text((x_base, y), str(text)[:30], fill=(200, 200, 200), font=self._font)
                y += 20

        # 转回 OpenCV 格式
        output = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return output

    # ==================== 工具方法 ====================

    def create_debug_window(self, name: str, frame: np.ndarray = None):
        """创建调试窗口（仅 GUI 模式有效）"""
        if not self.enable_gui:
            return
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if frame is not None:
            cv2.imshow(name, frame)

    def show_frame(self, name: str, frame: np.ndarray):
        """显示帧（仅 GUI 模式有效）"""
        if self.enable_gui and frame is not None:
            cv2.imshow(name, frame)

    def destroy(self):
        """销毁所有窗口"""
        if self.enable_gui:
            cv2.destroyAllWindows()
