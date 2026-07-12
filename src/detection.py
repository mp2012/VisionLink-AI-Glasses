"""
YOLO 实时障碍物检测模块
基于胸前深度相机（FOV）的常开视频流，进行高频低延迟障碍物检测

特性：
- 独立线程运行，不阻塞大模型推理和 TTS 播报
- 支持 Orbbec 深度相机真实距离测量 + bbox 位置估算双模式
- 分级预警（警告/危险）
- 播报冷却机制，避免重复提醒
- 异常重连与日志记录
"""
import time
import logging
import threading
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

import cv2
import numpy as np

from .platform import IS_JETSON
from .config import YOLO_CONFIG, DEPTH_CONFIG, YOLO_PAUSE_EVENT

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """单次检测结果"""
    timestamp: float = 0.0
    objects: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dangers: List[str] = field(default_factory=list)

    @property
    def has_alert(self) -> bool:
        return len(self.warnings) > 0 or len(self.dangers) > 0

    @property
    def alert_message(self) -> str:
        """生成播报消息"""
        messages = []
        if self.dangers:
            messages.append("危险！" + "；".join(self.dangers))
        if self.warnings:
            messages.append("注意，" + "；".join(self.warnings))
        return "。".join(messages)


# YOLO 类别名称映射（COCO 数据集）
COCO_CLASSES = {
    0: "人", 1: "自行车", 2: "汽车", 3: "摩托车",
    5: "公交车", 7: "卡车", 9: "交通灯", 11: "停止标志",
    13: "长椅", 14: "鸟", 15: "猫", 16: "狗",
    17: "马", 39: "瓶子", 44: "瓶子", 56: "椅子",
    60: "餐桌", 62: "电视", 63: "笔记本电脑", 67: "手机",
    73: "书本", 77: "手机",
}

# 障碍物危险等级分类（描述不含"前方"，位置由 _describe_position 提供）
OBSTACLE_LEVELS = {
    "danger": {0: "有人", 2: "有车", 3: "有摩托车", 5: "有公交车", 7: "有卡车"},
    "warning": {1: "有自行车", 13: "有长椅", 56: "有椅子"},
}


class YOLODetector:
    """
    YOLO 实时障碍物检测器
    独立线程运行，通过回调函数输出检测结果

    支持两种深度模式：
    1. Orbbec 真实深度：通过 depth_camera 参数传入 OrbbecDepthCamera 实例
    2. bbox 位置估算：仅使用 FOV RGB 摄像头，根据目标在画面中的位置估算距离

    Usage:
        # 纯 RGB 模式
        detector = YOLODetector(fov_camera, on_detect=handle_detection)

        # 深度相机模式
        detector = YOLODetector(fov_camera, depth_camera=orbbec_cam, on_detect=handle_detection)
        detector.start()
        ...
        detector.stop()
    """

    def __init__(self, camera_manager, depth_camera=None, on_detect=None, config: dict = None):
        """
        Args:
            camera_manager: FOV 摄像头管理器实例
            depth_camera: OrbbecDepthCamera 实例（可选，启用真实深度测量）
            on_detect: 检测回调函数 callback(DetectionResult)
            config: 检测配置字典，默认使用 YOLO_CONFIG
        """
        self.camera = camera_manager
        self.depth_camera = depth_camera
        self.on_detect = on_detect
        self.config = {**YOLO_CONFIG, **(config or {})}
        self._depth_config = DEPTH_CONFIG

        self._model = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_announce_time = 0.0
        self._latest_result = DetectionResult()

        # 统计
        self._frame_count = 0
        self._detect_count = 0

    @property
    def has_depth(self) -> bool:
        """是否启用了真实深度相机"""
        return self.depth_camera is not None and self.depth_camera.is_running()

    def _load_model(self) -> bool:
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
            model_path = self.config["model_path"]
            self._model = YOLO(model_path)
            logger.info(f"YOLO 模型已加载: {model_path}")
            return True
        except ImportError:
            logger.error("ultralytics 未安装，请执行: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            return False

    def start(self) -> bool:
        """启动检测线程"""
        if self._running:
            logger.warning("YOLO 检测器已在运行中")
            return False

        if not self._load_model():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._detect_loop, daemon=True, name="YOLO-Detect")
        self._thread.start()
        logger.info("YOLO 避障检测已启动")
        return True

    def stop(self):
        """停止检测线程"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("YOLO 避障检测已停止")

    def is_running(self) -> bool:
        return self._running

    @property
    def latest_result(self) -> DetectionResult:
        """获取最新检测结果"""
        with self._lock:
            return self._latest_result

    def get_stats(self) -> Dict[str, int]:
        """获取运行统计"""
        return {
            "frame_count": self._frame_count,
            "detect_count": self._detect_count,
        }

    def _detect_loop(self):
        """检测主循环（独立线程）"""
        detect_interval = self.config["detect_interval"]
        last_detect_time = 0.0

        while self._running:
            # 独占模式：VLM 推理进行时暂停 YOLO，释放 GPU 显存
            if not YOLO_PAUSE_EVENT.is_set():
                YOLO_PAUSE_EVENT.wait(timeout=0.5)
                continue

            try:
                # 读取 FOV 帧
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                self._frame_count += 1
                now = time.time()

                # 按间隔执行检测（控制检测频率）
                if now - last_detect_time < detect_interval:
                    continue

                last_detect_time = now
                result = self._detect_frame(frame)
                result.timestamp = now

                with self._lock:
                    self._latest_result = result
                    self._detect_count += 1

                # 回调通知
                if result.has_alert and self.on_detect:
                    if now - self._last_announce_time >= self.config["announce_cooldown"]:
                        self._last_announce_time = now
                        try:
                            self.on_detect(result)
                        except Exception as e:
                            logger.error(f"检测回调异常: {e}")

            except Exception as e:
                logger.error(f"YOLO 检测循环异常: {e}")
                time.sleep(0.1)

    def _detect_frame(self, frame) -> DetectionResult:
        """对单帧执行目标检测，融合深度数据"""
        result = DetectionResult()
        conf_threshold = self.config["confidence_threshold"]

        try:
            if self._model is None:
                return result

            # 获取最新深度图（后台线程持续缓存，非阻塞读取）
            depth_map = None
            if self.has_depth:
                depth_map, _ = self.depth_camera.get_frames()

            # YOLO 推理
            detections = self._model(frame, verbose=False)
            if not detections or len(detections) == 0:
                return result

            det = detections[0]
            if det.boxes is None:
                return result

            boxes = det.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = boxes.conf[i].item()

                if conf < conf_threshold:
                    continue

                # 获取边界框
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # 计算目标中心位置（归一化坐标）
                cx = (x1 + x2) / 2 / frame.shape[1]
                cy = (y1 + y2) / 2 / frame.shape[0]

                # 获取真实深度（mm）
                depth_mm = -1.0
                if depth_map is not None:
                    depth_mm = self.depth_camera.get_bbox_depth(depth_map, x1, y1, x2, y2)

                # 位置描述 + 距离描述
                position = self._describe_position(cx, cy, depth_mm)
                class_name = COCO_CLASSES.get(cls_id, f"物体({cls_id})")

                obj_info = {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "position": position,
                    "depth_mm": depth_mm,
                }
                result.objects.append(obj_info)

                # 分级预警（融合深度数据）
                alert_text = self._classify_alert(cls_id, position, depth_mm)
                if alert_text:
                    if cls_id in OBSTACLE_LEVELS.get("danger", {}):
                        result.dangers.append(alert_text)
                    elif cls_id in OBSTACLE_LEVELS.get("warning", {}):
                        result.warnings.append(alert_text)

        except Exception as e:
            logger.error(f"帧检测异常: {e}")

        return result

    def _describe_position(self, cx: float, cy: float, depth_mm: float = -1) -> str:
        """
        根据目标中心位置和深度描述方位

        Args:
            cx, cy: 目标中心归一化坐标 (0~1)
            depth_mm: 真实深度 (mm)，-1 表示不可用

        Returns:
            方位描述字符串，如 "正前方约2米处"
        """
        if cx < 0.35:
            h_pos = "左侧"
        elif cx > 0.65:
            h_pos = "右侧"
        else:
            h_pos = "正前方"

        if cy < 0.35:
            v_pos = "远处"
        elif cy > 0.65:
            v_pos = "近处"
        else:
            v_pos = "前方"

        # 如果深度数据可用，附加距离信息
        if depth_mm > 0:
            if depth_mm < 1000:
                dist_text = f"约{int(depth_mm / 100) * 100}毫米"
            else:
                dist_text = self._format_distance_zh(depth_mm)
            return f"{h_pos}{v_pos}{dist_text}"
        else:
            return f"{h_pos}{v_pos}"

    @staticmethod
    def _format_distance_zh(depth_mm: float) -> str:
        """
        将距离转为中文口语化表达，避免 TTS 误读
        例如：1100mm → "约一点一米"，1050mm → "约一米"，980mm → "约980毫米"
        """
        meters = depth_mm / 1000.0
        if meters < 1.0:
            return f"约{int(depth_mm / 100) * 100}毫米"

        # 保留一位小数，转换为中文口语数字
        m = round(meters, 1)
        int_part = int(m)
        frac_part = round((m - int_part) * 10)

        digits_cn = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
                      "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}

        if frac_part == 0:
            return f"约{''.join(digits_cn.get(c, c) for c in str(int_part))}米"
        else:
            int_str = ''.join(digits_cn.get(c, c) for c in str(int_part))
            frac_str = digits_cn.get(str(frac_part), str(frac_part))
            return f"约{int_str}点{frac_str}米"

    def _classify_alert(self, cls_id: int, position: str, depth_mm: float = -1) -> str:
        """
        根据类别、位置和深度生成预警文本

        Args:
            cls_id: YOLO 类别 ID
            position: 方位描述字符串
            depth_mm: 真实深度 (mm)

        Returns:
            预警文本字符串
        """
        # 如果深度数据可用，根据距离调整预警等级
        if depth_mm > 0:
            danger_dist = self._depth_config.get("danger_distance_mm", 500)
            warning_dist = self._depth_config.get("warning_distance_mm", 1500)

            if depth_mm < danger_dist:
                # 进入危险范围，强制提升为 danger
                if cls_id in OBSTACLE_LEVELS["danger"]:
                    return f"{position}{OBSTACLE_LEVELS['danger'][cls_id]}，距离很近！"
                elif cls_id in OBSTACLE_LEVELS["warning"]:
                    return f"{position}{OBSTACLE_LEVELS['warning'][cls_id]}，距离很近！"
            elif depth_mm < warning_dist:
                if cls_id in OBSTACLE_LEVELS["danger"]:
                    return f"{position}{OBSTACLE_LEVELS['danger'][cls_id]}"
                elif cls_id in OBSTACLE_LEVELS["warning"]:
                    return f"{position}{OBSTACLE_LEVELS['warning'][cls_id]}"

            # 超出预警范围，静默
            return ""

        # 无深度数据时，使用原有的位置估算逻辑
        if cls_id in OBSTACLE_LEVELS["danger"]:
            return f"{position}{OBSTACLE_LEVELS['danger'][cls_id]}"
        if cls_id in OBSTACLE_LEVELS["warning"]:
            return f"{position}{OBSTACLE_LEVELS['warning'][cls_id]}"
        return ""

    def annotate_frame(self, frame) -> np.ndarray:
        """
        在帧上绘制检测结果（用于可视化调试）

        Args:
            frame: 原始帧

        Returns:
            标注后的帧
        """
        result = self.latest_result
        annotated = frame.copy()

        for obj in result.objects:
            x1, y1, x2, y2 = obj["bbox"]
            class_name = obj["class_name"]
            conf = obj["confidence"]

            # 根据危险等级选择颜色
            if obj["class_id"] in OBSTACLE_LEVELS["danger"]:
                color = (0, 0, 255)  # 红色
            elif obj["class_id"] in OBSTACLE_LEVELS["warning"]:
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 255, 0)  # 绿色

            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{class_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return annotated
