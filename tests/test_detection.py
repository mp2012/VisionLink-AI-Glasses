"""
测试 src/detection.py — 检测逻辑（不含模型加载）
"""
import pytest
from src.detection import (
    DetectionResult, YOLODetector, COCO_CLASSES, OBSTACLE_LEVELS
)


class TestDetectionResult:
    """DetectionResult 数据类测试"""

    def test_empty_result_no_alert(self):
        r = DetectionResult()
        assert not r.has_alert
        assert r.alert_message == ""

    def test_warning_only(self):
        r = DetectionResult(warnings=["前方有自行车"])
        assert r.has_alert
        assert "注意" in r.alert_message
        assert "前方有自行车" in r.alert_message

    def test_danger_only(self):
        r = DetectionResult(dangers=["前方有人"])
        assert r.has_alert
        assert "危险" in r.alert_message
        assert "前方有人" in r.alert_message

    def test_mixed_alerts(self):
        r = DetectionResult(
            warnings=["左侧有椅子"],
            dangers=["正前方有车"]
        )
        assert r.has_alert
        assert "危险" in r.alert_message
        assert "注意" in r.alert_message
        # 危险信息应该先于警告
        msg = r.alert_message
        assert msg.index("危险") < msg.index("注意")

    def test_multiple_dangers(self):
        r = DetectionResult(dangers=["前方有人", "右侧有车"])
        assert "危险" in r.alert_message
        assert "前方有人" in r.alert_message
        assert "右侧有车" in r.alert_message

    def test_multiple_warnings(self):
        r = DetectionResult(warnings=["左侧有椅子", "右侧有自行车"])
        assert "左侧有椅子" in r.alert_message
        assert "右侧有自行车" in r.alert_message

    def test_objects_list(self):
        r = DetectionResult(objects=[{"class_name": "人", "confidence": 0.9}])
        assert len(r.objects) == 1
        assert r.objects[0]["class_name"] == "人"

    def test_timestamp_default(self):
        r = DetectionResult()
        assert r.timestamp == 0.0

    def test_timestamp_set(self):
        r = DetectionResult(timestamp=12345.67)
        assert r.timestamp == 12345.67


class TestCOCOClasses:
    """COCO 类别映射测试"""

    def test_known_classes(self):
        assert COCO_CLASSES[0] == "人"
        assert COCO_CLASSES[2] == "汽车"
        assert COCO_CLASSES[3] == "摩托车"

    def test_all_danger_classes_exist(self):
        """危险类别定义中的 class_id 必须在 COCO_CLASSES 中存在"""
        for level, mapping in OBSTACLE_LEVELS.items():
            for cls_id in mapping:
                assert cls_id in COCO_CLASSES, \
                    f"class_id={cls_id} ({mapping[cls_id]}) 不在 COCO_CLASSES 中"

    def test_obstacle_levels_structure(self):
        """OBSTACLE_LEVELS 必须有 danger 和 warning 两级"""
        assert "danger" in OBSTACLE_LEVELS
        assert "warning" in OBSTACLE_LEVELS


class TestPositionDescription:
    """方位描述测试"""

    def test_center_position(self):
        """目标在画面中心"""
        detector = _make_detector()
        pos = detector._describe_position(0.5, 0.5, -1)
        assert "正前方" in pos

    def test_left_position(self):
        """目标在画面左侧"""
        detector = _make_detector()
        pos = detector._describe_position(0.2, 0.5, -1)
        assert "左侧" in pos

    def test_right_position(self):
        """目标在画面右侧"""
        detector = _make_detector()
        pos = detector._describe_position(0.8, 0.5, -1)
        assert "右侧" in pos

    def test_boundary_left(self):
        """边界测试: cx=0.35 为左侧"""
        detector = _make_detector()
        pos = detector._describe_position(0.34, 0.5, -1)
        assert "左侧" in pos

    def test_boundary_right(self):
        """边界测试: cx=0.65 为右侧"""
        detector = _make_detector()
        pos = detector._describe_position(0.66, 0.5, -1)
        assert "右侧" in pos

    def test_near_position(self):
        """目标在画面下侧（近处）"""
        detector = _make_detector()
        pos = detector._describe_position(0.5, 0.8, -1)
        assert "近处" in pos

    def test_far_position(self):
        """目标在画面上侧（远处）"""
        detector = _make_detector()
        pos = detector._describe_position(0.5, 0.2, -1)
        assert "远处" in pos

    def test_with_depth(self):
        """有深度数据时包含距离信息"""
        detector = _make_detector()
        pos = detector._describe_position(0.5, 0.5, 800)
        assert "800毫米" in pos or "约" in pos


class TestDistanceFormatting:
    """距离格式化测试"""

    def test_millimeters(self):
        """小于 1 米的距离显示毫米"""
        result = YOLODetector._format_distance_zh(500)
        assert "毫米" in result or "500" in result

    def test_exact_one_meter(self):
        """正好 1 米"""
        result = YOLODetector._format_distance_zh(1000)
        assert result is not None

    def test_more_than_one_meter(self):
        """大于 1 米"""
        result = YOLODetector._format_distance_zh(1500)
        assert "米" in result

    def test_very_large_distance(self):
        """很大距离"""
        result = YOLODetector._format_distance_zh(10000)
        assert "米" in result

    def test_zero_distance(self):
        """零距离"""
        result = YOLODetector._format_distance_zh(0)
        assert result is not None


class TestAlertClassification:
    """告警分级测试"""

    def test_danger_class_near(self):
        """危险类别 + 近距离"""
        detector = _make_detector()
        alert = detector._classify_alert(0, "正前方", 300)  # person, 300mm
        assert alert != "" and "距离很近" in alert

    def test_danger_class_far(self):
        """危险类别 + 远距离"""
        detector = _make_detector()
        alert = detector._classify_alert(0, "正前方", 2000)  # person, 2000mm
        assert alert == ""  # 超出预警范围

    def test_warning_class_near(self):
        """预警类别 + 近距离（升级为 danger 级描述）"""
        detector = _make_detector()
        alert = detector._classify_alert(1, "右侧", 300)  # bicycle, 300mm
        assert alert != "" and "距离很近" in alert

    def test_no_depth_fallback(self):
        """无深度相机时，仅基于类别判断"""
        detector = _make_detector()
        alert = detector._classify_alert(0, "正前方", -1)  # person, no depth
        assert alert != ""

    def test_unknown_class(self):
        """非障碍物类别返回空"""
        detector = _make_detector()
        alert = detector._classify_alert(999, "正前方", 500)
        assert alert == ""

    def test_warning_range(self):
        """预警距离范围内"""
        detector = _make_detector()
        alert = detector._classify_alert(2, "正前方", 1000)  # car, 1000mm
        # 在 danger(500) 和 warning(1500) 之间
        assert alert != ""


class TestYOLOConfig:
    """YOLO 检测器配置测试"""

    def test_default_config_complete(self):
        detector = _make_detector()
        assert detector.config["confidence_threshold"] == 0.5
        assert detector.config["detect_interval"] > 0

    def test_custom_config_override(self):
        """自定义配置可覆盖默认值"""
        detector = _make_detector(config={"confidence_threshold": 0.8})
        assert detector.config["confidence_threshold"] == 0.8

    def test_has_depth_without_camera(self):
        """没有深度相机时应返回 False"""
        detector = _make_detector()
        assert not detector.has_depth

    def test_stats_initial(self):
        """初始统计数据"""
        detector = _make_detector()
        stats = detector.get_stats()
        assert stats["frame_count"] == 0
        assert stats["detect_count"] == 0


def _make_detector(config: dict = None):
    """创建不带摄像头的 YOLODetector 实例用于测试"""

    class FakeCamera:
        def read(self):
            import numpy as np
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

    return YOLODetector(FakeCamera(), on_detect=lambda r: None, config=config)
