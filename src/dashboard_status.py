"""
VisionLink 综合监控仪表板 —— 状态容器 + 硬件遥测

线程安全的单例状态容器，提供：
- 应用状态追踪（模式/YOLO/深度相机/Ollama）
- 事件日志环形缓冲区（推理/检测/TTS）
- 硬件遥测（jtop GPU+CPU+温度，psutil 降级兜底）

使用方式:
    from dashboard_status import system_status

    # 启动硬件监控（仅需一次）
    system_status.start_hardware_monitor()

    # 各模块打点
    system_status.set_mode(1, "障碍物检测")
    system_status.log_inference("眼前是一台电脑", success=True, latency_ms=3400)
    system_status.log_tts(priority=0, text="右侧800mm有人", engine="piper")

    # Flask 路由读取快照
    data = system_status.snapshot()

设计原则：
    - 零依赖崩溃：jtop/psutil 不可用时静默降级
    - 最小开销：锁粒度只覆盖 dict/list 操作
    - 环形缓冲：每个日志类型最多 50 条，防止内存泄漏
"""

import threading
import time
import logging
import os
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# 环形缓冲区容量
_MAX_EVENTS = 50
# 硬件轮询间隔（秒）
_HW_POLL_INTERVAL = 2.0


class SystemStatus:
    """线程安全的状态容器单例"""

    def __init__(self):
        self._lock = threading.RLock()
        self._reset()

    def _reset(self):
        """重置所有状态字段"""
        # ── 应用状态 ──
        self._mode: int = 0
        self._mode_name: str = "--"
        self._lang: str = "zh"
        self._app_state: str = "idle"
        self._auto_enabled: bool = False

        # ── 子模块连接状态 ──
        self._yolo_enabled: bool = False
        self._yolo_running: bool = False
        self._depth_camera_ok: bool = False
        self._ollama_connected: bool = False
        self._dual_camera: bool = False

        # ── 推理日志 ──
        self._inference_logs: List[Dict[str, Any]] = []
        self._last_inference_text: str = ""
        self._last_inference_time: float = 0.0

        # ── 检测日志 ──
        self._detection_logs: List[Dict[str, Any]] = []
        self._last_detection_text: str = ""
        self._last_detection_time: float = 0.0
        self._last_detection_priority: int = -1

        # ── TTS 日志 ──
        self._tts_logs: List[Dict[str, Any]] = []
        self._tts_speaking: bool = False
        self._tts_priority: Optional[int] = None

        # ── 硬件遥测 ──
        self._hw_thread: Optional[threading.Thread] = None
        self._hw_running: bool = False
        self._gpu_percent: float = 0.0
        self._cpu_percent: float = 0.0
        self._mem_percent: float = 0.0
        self._temperature: float = 0.0
        self._hw_available: bool = False

        # ── 最近触发快照 ──
        self._last_snapshot_path: str = ""
        self._last_snapshot_time: float = 0.0
        self._last_snapshot_mode: int = 0

    # ==================== 应用状态打点 ====================

    def set_mode(self, mode_idx: int, mode_name: str = ""):
        """模式切换时调用"""
        with self._lock:
            self._mode = mode_idx
            self._mode_name = mode_name or f"模式{mode_idx}"

    def set_lang(self, lang: str):
        with self._lock:
            self._lang = lang

    def set_app_state(self, state: str):
        with self._lock:
            self._app_state = state

    def set_auto(self, enabled: bool):
        with self._lock:
            self._auto_enabled = enabled

    def set_yolo_enabled(self, enabled: bool):
        with self._lock:
            self._yolo_enabled = enabled

    def set_yolo_running(self, running: bool):
        with self._lock:
            self._yolo_running = running

    def set_depth_camera_ok(self, ok: bool):
        with self._lock:
            self._depth_camera_ok = ok

    def set_ollama_connected(self, connected: bool):
        with self._lock:
            self._ollama_connected = connected

    def set_dual_camera(self, dual: bool):
        with self._lock:
            self._dual_camera = dual

    # ==================== 事件日志打点 ====================

    def log_inference(self, text: str, success: bool, latency_ms: float = 0):
        """
        推理完成后调用。

        Args:
            text: 推理结果文本（空字符串表示失败）
            success: 是否成功
            latency_ms: 推理耗时（毫秒）
        """
        entry = {
            "time": time.time(),
            "text": text[:200] if text else "",
            "success": success,
            "latency_ms": round(latency_ms),
        }
        with self._lock:
            self._inference_logs.append(entry)
            if len(self._inference_logs) > _MAX_EVENTS:
                self._inference_logs = self._inference_logs[-_MAX_EVENTS:]
            if success and text:
                self._last_inference_text = text
                self._last_inference_time = entry["time"]

    def log_detection(self, text: str, priority: int):
        """
        YOLO 检测到障碍物时调用。

        Args:
            text: 播报文本
            priority: 0=P0 危险, 1=P1 预警
        """
        if not text:
            return
        entry = {
            "time": time.time(),
            "text": text[:300],
            "priority": priority,
        }
        with self._lock:
            self._detection_logs.append(entry)
            if len(self._detection_logs) > _MAX_EVENTS:
                self._detection_logs = self._detection_logs[-_MAX_EVENTS:]
            self._last_detection_text = text
            self._last_detection_time = entry["time"]
            self._last_detection_priority = priority

    def log_tts(self, priority: int, text: str = "", engine: str = ""):
        """
        TTS 开始播放时调用。

        Args:
            priority: 优先级 (0=P0 紧急, 1=P1 预警, 2=P2 系统, 3=P3 常规)
            text: 播报文本
            engine: TTS 引擎名称 (piper/espeak/edge)
        """
        entry = {
            "time": time.time(),
            "priority": priority,
            "text": text[:120] if text else "",
            "engine": engine,
        }
        with self._lock:
            self._tts_logs.append(entry)
            if len(self._tts_logs) > _MAX_EVENTS:
                self._tts_logs = self._tts_logs[-_MAX_EVENTS:]
            self._tts_speaking = True
            self._tts_priority = priority

    def set_tts_idle(self):
        """TTS 播放完成时调用"""
        with self._lock:
            self._tts_speaking = False
            self._tts_priority = None

    def set_last_snapshot(self, path: str, mode: int):
        """触发推理拍快照时调用（在保存快照后立即调用）

        Args:
            path: 快照文件绝对路径
            mode: 触发时的模式编号 (1-5)
        """
        with self._lock:
            self._last_snapshot_path = path
            self._last_snapshot_time = time.time()
            self._last_snapshot_mode = mode

    # ==================== 快照读取 ====================

    def snapshot(self) -> Dict[str, Any]:
        """
        返回完整状态快照（供 /api/dashboard 使用）。

        注意：返回的 dict 是深拷贝后的独立副本，
        前端可安全读取，不受后续状态更新影响。
        """
        with self._lock:
            return {
                "ts": time.time(),

                # 应用状态
                "mode_idx": self._mode,
                "mode_name": self._mode_name,
                "lang": self._lang,
                "state": self._app_state,
                "auto_enabled": self._auto_enabled,

                # 子模块
                "yolo_enabled": self._yolo_enabled,
                "yolo_running": self._yolo_running,
                "depth_camera_ok": self._depth_camera_ok,
                "ollama_connected": self._ollama_connected,
                "dual_camera": self._dual_camera,

                # TTS 实时状态
                "tts_speaking": self._tts_speaking,
                "tts_priority": self._tts_priority,

                # 最新事件
                "last_detection_text": self._last_detection_text,
                "last_detection_time": self._last_detection_time,
                "last_detection_priority": self._last_detection_priority,
                "last_inference_text": self._last_inference_text,
                "last_inference_time": self._last_inference_time,

                # 事件日志（最近 20 条）
                "tts_logs": list(self._tts_logs[-20:]),
                "inf_logs": list(self._inference_logs[-10:]),
                "det_logs": list(self._detection_logs[-10:]),

                # 硬件遥测
                "gpu_percent": self._gpu_percent,
                "cpu_percent": self._cpu_percent,
                "mem_percent": self._mem_percent,
                "temperature": self._temperature,
                "hw_available": self._hw_available,

                # 最近触发快照
                "last_snapshot_url": (
                    f"/snapshot/{os.path.basename(self._last_snapshot_path)}"
                    if self._last_snapshot_path else None
                ),
                "last_snapshot_time": self._last_snapshot_time,
                "last_snapshot_mode": self._last_snapshot_mode,
            }

    # ==================== 硬件遥测 ====================

    def start_hardware_monitor(self):
        """启动硬件监控后台线程（幂等，重复调用不重复启动）"""
        if self._hw_running:
            return
        self._hw_running = True
        self._hw_thread = threading.Thread(
            target=self._hw_loop, daemon=True, name="HWMonitor"
        )
        self._hw_thread.start()
        logger.info("硬件遥测已启动")

    def stop_hardware_monitor(self):
        """停止硬件监控线程"""
        self._hw_running = False
        if self._hw_thread and self._hw_thread.is_alive():
            self._hw_thread.join(timeout=3.0)
        self._hw_thread = None

    def _hw_loop(self):
        """硬件轮询主循环（后台 daemon 线程）"""
        jtop = None
        jtop_ok = False

        # 尝试初始化 jtop（Jetson 专用）
        try:
            from jtop import jtop as Jtop
            jtop = Jtop()
            jtop.start()
            # 等待首次数据就绪
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if jtop.ok():
                    jtop_ok = True
                    break
                time.sleep(0.5)
            if jtop_ok:
                logger.info("jtop 硬件遥测已连接")
            else:
                logger.info("jtop 连接超时，使用 psutil 降级")
                try:
                    jtop.close()
                except Exception:
                    pass
                jtop = None
        except ImportError:
            logger.info("jtop 未安装，使用 psutil 降级监控")
        except Exception as e:
            logger.info(f"jtop 初始化失败 ({e})，使用 psutil 降级")
            jtop = None

        self._hw_available = jtop_ok

        _hw_err_reported = 0  # 错误限频计数器

        try:
            while self._hw_running:
                try:
                    if jtop and jtop_ok:
                        self._read_jtop(jtop)
                        self._hw_available = True
                    elif jtop:
                        # jtop 对象存在但 ok() 异常 → 尝试恢复
                        try:
                            if jtop.ok():
                                jtop_ok = True
                                self._hw_available = True
                                self._read_jtop(jtop)
                            else:
                                self._read_psutil()
                                self._hw_available = False
                        except Exception:
                            self._read_psutil()
                            self._hw_available = False
                    else:
                        self._read_psutil()
                        self._hw_available = False
                except Exception as e:
                    _hw_err_reported += 1
                    if _hw_err_reported <= 3:
                        logger.warning(f"硬件采样异常 [{_hw_err_reported}]: {e}")
                    elif _hw_err_reported % 30 == 0:
                        logger.warning(f"硬件采样异常 [{_hw_err_reported}]: {e}")

                time.sleep(_HW_POLL_INTERVAL)
        finally:
            if jtop:
                try:
                    jtop.close()
                except Exception:
                    pass

    def _read_jtop(self, jtop) -> None:
        """从 jtop 读取 Jetson 硬件指标

        jtop.stats 真实键名（Orin NX/AGX）:
            GPU: float            → 使用率百分比 (0-100)
            CPU1..CPUn: int       → 每个核心使用率 (0-100)
            RAM: float            → 内存占用比例 (0.0-1.0)，需 ×100 转百分比
            Temp gpu: float       → GPU 温度 (°C)
            Temp cpu: float       → CPU 温度 (°C)

        兼容不同 jtop 版本中 CPU key 的变体（如 CPU1, CPU1_A78, CPU1_CORTEX-A78）。
        """
        stats = jtop.stats
        if not stats:
            return

        # CPU: 匹配 "CPU" 开头 + 数字 的键，兼容后缀变体
        cpu_vals = []
        for k, v in stats.items():
            if isinstance(k, str) and k.startswith("CPU"):
                suffix = k[3:]
                # 提取开头的数字部分：CPU1 → 1, CPU1_A78 → 1
                if suffix and suffix[0].isdigit():
                    cpu_vals.append(float(v))

        gpu_val = float(stats.get("GPU", 0))
        if gpu_val == 0:
            # 部分 jtop 版本使用 "GPU0" 或 "GPUGPU"
            gpu_val = float(stats.get("GPU0", 0))

        # 温度：优先 GPU 温度，fallback CPU
        gpu_temp = stats.get("Temp gpu")
        cpu_temp = stats.get("Temp cpu")
        temp_val = float(
            gpu_temp if gpu_temp is not None and gpu_temp > 0
            else (cpu_temp if cpu_temp is not None and cpu_temp > 0 else 0)
        )

        # 内存：兼容 float (0.0-1.0) 和 dict {"used":..., "total":...} 两种格式
        ram = stats.get("RAM")
        if isinstance(ram, (int, float)):
            mem_val = float(ram) * 100.0
        elif isinstance(ram, dict):
            used = ram.get("used", 0)
            total = ram.get("total", 1)
            mem_val = (used / total) * 100.0 if total > 0 else 0.0
        else:
            mem_val = 0.0

        with self._lock:
            self._gpu_percent = gpu_val
            if cpu_vals:
                self._cpu_percent = sum(cpu_vals) / len(cpu_vals)
            self._temperature = temp_val
            self._mem_percent = mem_val

        # 前 3 次成功采样打印调试日志
        _sample_count = getattr(self, '_hw_sample_count', 0) + 1
        self._hw_sample_count = _sample_count
        if _sample_count <= 3:
            logger.info(
                f"硬件遥测 #{_sample_count}: "
                f"GPU={gpu_val:.0f}% CPU={self._cpu_percent:.0f}% "
                f"MEM={mem_val:.0f}% TEMP={temp_val:.0f}°C "
                f"(CPU keys: {len(cpu_vals)})"
            )

    def _read_psutil(self) -> None:
        """psutil 降级读取（通用 Linux）

        注意: 所有可能阻塞的 I/O 调用放在锁外，锁内仅赋值。
        """
        try:
            import psutil
        except ImportError:
            return

        # 阻塞调用放在锁外
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent

        temp = 0.0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for entries in temps.values():
                    if entries:
                        temp = float(entries[0].current)
                        break
        except Exception:
            pass

        with self._lock:
            self._cpu_percent = cpu
            self._mem_percent = mem
            if temp > 0:
                self._temperature = temp

    # ==================== 生命周期 ====================

    def shutdown(self):
        """程序退出时调用"""
        self.stop_hardware_monitor()
        logger.info("状态容器已关闭")


# 模块级单例
system_status = SystemStatus()
