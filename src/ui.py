"""
UI 绘制模块
Windows: PIL 中文渲染 + OpenCV 半透明侧面板
Jetson 有显示器: 简化版 UI
Jetson 无头: 空实现（Null Object Pattern），仅日志输出
"""
import logging
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .platform import IS_JETSON, IS_WINDOWS, HAS_DISPLAY
from .config import (
    UI_PANEL_WIDTH, UI_ALPHA, FONT_PATHS, FONT_SIZES,
    STATE_IDLE, STATE_CAPTURE, STATE_LISTEN, STATE_INFER, STATE_TTS,
)
from .prompts import MODE_NAME_LIST, STATE_NAME_LIST, GUIDE_TEXT

logger = logging.getLogger(__name__)


class UIManager:
    """跨平台 UI 管理器"""

    def __init__(self):
        self.enable_gui = IS_WINDOWS or HAS_DISPLAY
        self.fonts = {}
        if self.enable_gui:
            self._load_fonts()
        self._current_state = STATE_IDLE
        self._current_mode = 1
        self._auto_enabled = False
        self._voice_lang = "zh"

    def _load_fonts(self):
        """加载中文字体，依次尝试备选路径"""
        for font_path in FONT_PATHS:
            try:
                sizes = FONT_SIZES
                self.fonts = {
                    sizes[0]: ImageFont.truetype(font_path, sizes[0]),
                    sizes[1]: ImageFont.truetype(font_path, sizes[1]),
                    sizes[2]: ImageFont.truetype(font_path, sizes[2]),
                }
                logger.info(f"字体加载成功: {font_path}")
                return
            except Exception:
                continue
        logger.error("所有中文字体加载失败，UI 将不可用")
        self.enable_gui = False

    def update_state(self, state: int, mode: int, auto: bool, lang: str):
        """更新 UI 需要显示的状态信息"""
        self._current_state = state
        self._current_mode = mode
        self._auto_enabled = auto
        self._voice_lang = lang

    def draw_panel(self, frame):
        """绘制侧边信息面板，无头模式直接返回原图"""
        if not self.enable_gui:
            return frame

        h, w = frame.shape[:2]
        panel_w = UI_PANEL_WIDTH

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, UI_ALPHA, frame, 1 - UI_ALPHA, 0)

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        font_l = self.fonts.get(FONT_SIZES[0])
        font_m = self.fonts.get(FONT_SIZES[1])
        font_s = self.fonts.get(FONT_SIZES[2])
        if not all([font_l, font_m, font_s]):
            return frame

        lang = self._voice_lang
        y = 30

        # 状态指示
        color_map = {
            STATE_IDLE: (0, 220, 0),
            STATE_CAPTURE: (220, 150, 0),
            STATE_LISTEN: (255, 100, 0),
            STATE_INFER: (160, 0, 220),
            STATE_TTS: (0, 100, 255),
        }
        draw.rectangle([10, y - 15, 30, y + 5], fill=color_map.get(self._current_state, (128, 128, 128)))
        status_txt = STATE_NAME_LIST[lang][self._current_state]
        prefix = "状态：" if lang == "zh" else "Status: "
        draw.text((45, y - 12), f"{prefix}{status_txt}", font=font_l, fill=(255, 255, 255))
        y += 45

        # 当前模式
        mode_txt = MODE_NAME_LIST[lang][self._current_mode - 1]
        prefix = "模式 " if lang == "zh" else "Mode "
        draw.text((15, y - 12), f"{prefix}{self._current_mode}：{mode_txt}", font=font_m, fill=(0, 255, 255))
        y += 40

        # 自动模式
        if self._auto_enabled:
            auto_txt = "自动模式：已开启(按M关闭)" if lang == "zh" else "Auto Agent: ON (M=OFF)"
            auto_color = (0, 255, 0)
        else:
            auto_txt = "自动模式：已关闭(按M开启)" if lang == "zh" else "Auto Agent: OFF (M=ON)"
            auto_color = (120, 120, 120)
        draw.text((15, y - 12), auto_txt, font=font_m, fill=auto_color)
        y += 40

        # 语种
        lang_txt = "当前语种：中文" if lang == "zh" else "Language: English"
        draw.text((15, y - 12), lang_txt, font=font_m, fill=(255, 200, 0))
        y += 45

        # 操作指引
        g = GUIDE_TEXT[lang]
        draw.text((15, y - 12), g["title"], font=font_m, fill=(255, 255, 255))
        y += 35
        for key_name in ["key1", "key2", "key3", "key4", "key5", "key6"]:
            color = (255, 255, 255)
            if key_name == "key3":
                color = (255, 200, 0)
            elif key_name == "key4":
                color = (0, 255, 0)
            elif key_name == "key5":
                color = (255, 0, 0)
            draw.text((15, y - 10), g[key_name], font=font_s, fill=color)
            y += 30
        y += 20

        draw.text((15, y - 10), g["pipe"], font=font_s, fill=(200, 200, 200))
        y = h - 25
        draw.text((15, y - 10), "VisionLink | Edge AI", font=font_s, fill=(150, 150, 150))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def show(self, window_name: str, frame):
        """显示画面，无头模式跳过"""
        if self.enable_gui:
            cv2.imshow(window_name, frame)

    def create_window(self, name: str, width: int, height: int):
        """创建预览窗口，无头模式跳过"""
        if self.enable_gui:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, width, height)

    def destroy(self):
        if self.enable_gui:
            cv2.destroyAllWindows()
