"""
测试 src/config.py — 配置完整性
"""
import pytest


class TestConfig:
    """配置值验证"""

    def test_yolo_config_has_required_keys(self):
        """YOLO_CONFIG 必须包含所有必要字段"""
        from src.config import YOLO_CONFIG
        required = [
            "model_path", "engine_path", "confidence_threshold",
            "nms_threshold", "detect_classes", "detect_interval",
            "depth_warning_distance", "depth_danger_distance", "announce_cooldown"
        ]
        for key in required:
            assert key in YOLO_CONFIG, f"YOLO_CONFIG 缺少字段: {key}"

    def test_yolo_config_values_range(self):
        """YOLO_CONFIG 值应在合理范围内"""
        from src.config import YOLO_CONFIG
        assert 0 < YOLO_CONFIG["confidence_threshold"] <= 1.0
        assert 0 < YOLO_CONFIG["nms_threshold"] <= 1.0
        assert 0 < YOLO_CONFIG["detect_interval"] <= 1.0
        assert 0.01 <= YOLO_CONFIG["announce_cooldown"] <= 60
        assert YOLO_CONFIG["depth_danger_distance"] < YOLO_CONFIG["depth_warning_distance"]

    def test_tts_priority_order(self):
        """TTS 优先级常量：数字越小优先级越高"""
        from src.config import (
            TTS_PRIORITY_EMERGENCY, TTS_PRIORITY_WARNING,
            TTS_PRIORITY_SYSTEM, TTS_PRIORITY_NORMAL,
        )
        assert TTS_PRIORITY_EMERGENCY == 0
        assert TTS_PRIORITY_WARNING == 1
        assert TTS_PRIORITY_SYSTEM == 2
        assert TTS_PRIORITY_NORMAL == 3
        # 确保优先级顺序正确
        priorities = [
            TTS_PRIORITY_EMERGENCY,
            TTS_PRIORITY_WARNING,
            TTS_PRIORITY_SYSTEM,
            TTS_PRIORITY_NORMAL,
        ]
        assert priorities == sorted(priorities), "优先级应该按 0,1,2,3 递增"

    def test_state_constants(self):
        """状态常量不重叠"""
        from src.config import STATE_IDLE, STATE_INFER, STATE_TTS
        states = [STATE_IDLE, STATE_INFER, STATE_TTS]
        assert len(states) == len(set(states)), "状态值不应重复"

    def test_mode_names_count(self):
        """MODE_NAMES 应该是 5 个模式"""
        from src.config import MODE_NAMES
        assert len(MODE_NAMES) == 5

    def test_depth_config_keys(self):
        """DEPTH_CONFIG 必须有完整的距离配置"""
        from src.config import DEPTH_CONFIG
        assert "warning_distance_mm" in DEPTH_CONFIG
        assert "danger_distance_mm" in DEPTH_CONFIG
        assert DEPTH_CONFIG["danger_distance_mm"] < DEPTH_CONFIG["warning_distance_mm"]

    def test_camera_config_structure(self):
        """摄像头配置结构一致"""
        from src.config import POV_CAMERA_CONFIG, FOV_CAMERA_CONFIG
        shared_keys = {"cam_id", "backend", "width", "height", "framerate"}
        for key in shared_keys:
            assert key in POV_CAMERA_CONFIG, f"POV 缺少 {key}"
            assert key in FOV_CAMERA_CONFIG, f"FOV 缺少 {key}"

    def test_yolo_pause_event_initial(self):
        """YOLO_PAUSE_EVENT 初始状态为 set（允许运行）"""
        from src.config import YOLO_PAUSE_EVENT
        assert YOLO_PAUSE_EVENT.is_set(), "YOLO 避障初始应为运行状态"

    def test_font_paths_not_empty(self):
        """字体路径列表不应为空"""
        from src.config import FONT_PATHS
        assert len(FONT_PATHS) > 0

    def test_sound_effects_has_shutter(self):
        """音效配置必须包含快门音"""
        from src.config import SOUND_EFFECTS
        assert "shutter" in SOUND_EFFECTS

    def test_ai_image_size_reasonable(self):
        """AI 图片尺寸合理"""
        from src.config import AI_IMAGE_SIZE
        assert 64 <= AI_IMAGE_SIZE <= 1024, f"AI_IMAGE_SIZE={AI_IMAGE_SIZE} 不在合理范围"

    def test_engine_path_default(self):
        """TensorRT 引擎初始设置正确"""
        from src.config import YOLO_ENGINE_PATH
        assert YOLO_ENGINE_PATH is not None
        assert YOLO_ENGINE_PATH.endswith(".engine")
