"""
核心控制中枢 - Agent
负责：状态机管理、按键触发/自动模式交互、场景去重、YOLO 避障回调

架构说明：
- 自动模式：定时扫描场景 → 任务分配 → 去重静默 → 异步执行
- 手动模式：按键触发单次推理 + TTS 播报
- YOLO 回调：接收避障检测结果 → 分级播报
"""
import os
import re
import time
import logging
import threading
from typing import Optional, Dict, Any, Callable

import cv2

from .config import (
    AGENT_SCAN_INTERVAL, BROADCAST_COOLDOWN, LONG_TIME_LIMIT,
    MAX_CONTEXT_ROUND, STATE_IDLE, STATE_INFER, STATE_TTS,
    MODE_NAMES, SNAPSHOT_DIR, INFER_LOG_DIR, SPACE_DEBOUNCE,
    YOLO_PAUSE_EVENT,
    TTS_PRIORITY_EMERGENCY, TTS_PRIORITY_WARNING,
    TTS_PRIORITY_SYSTEM, TTS_PRIORITY_NORMAL,
)
from .prompts import AGENT_PROMPT, TASK_PLAN_PROMPT, TIP_VOICE, PROMPT_LIB, MODE_NAME_LIST

logger = logging.getLogger(__name__)


class Agent:
    """
    核心控制中枢
    统一管理：状态机、自动/手动模式、推理调度、YOLO 避障回调

    Usage:
        agent = Agent(inference_engine, tts_engine, camera_manager)
        agent.set_mode(2)  # 文字识别模式
        agent.handle_trigger(frame)  # 手动触发推理
    """

    def __init__(self, inference_engine, tts_engine, camera_manager=None):
        self.infer = inference_engine
        self.tts = tts_engine
        self.camera = camera_manager

        # 状态机
        self._state = STATE_IDLE
        self._lock = threading.Lock()

        # 模式管理
        self._current_mode = 1  # 默认障碍物检测
        self._voice_lang = "zh"

        # 自动模式
        self._auto_enabled = False
        self._last_scan_time = 0.0
        self._last_broadcast = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self._last_tasks = []
        self._same_count = 0

        # 语音交互上下文
        self._context_history = []
        self._mic_listening = False

        # 防抖
        self._last_trigger_time = 0.0

        # YOLO 避障
        self._yolo_enabled = False
        self._yolo_last_announce = 0.0
        self._last_detection_text = ""   # 供 Web 仪表板读取
        self._last_detection_time = 0.0

        # VLM 推理
        self._last_inference_text = ""   # 供 Web 仪表板读取
        self._last_inference_time = 0.0

        # 回调注册
        self._callbacks: Dict[str, Callable] = {}

    # ==================== 属性 ====================

    @property
    def state(self) -> int:
        return self._state

    @property
    def current_mode(self) -> int:
        return self._current_mode

    @property
    def mode_name(self) -> str:
        fallback = "未知" if self._voice_lang == "zh" else "Unknown"
        return MODE_NAME_LIST[self._voice_lang][self._current_mode - 1] if 1 <= self._current_mode <= 5 else fallback

    @property
    def auto_enabled(self) -> bool:
        return self._auto_enabled

    @property
    def yolo_enabled(self) -> bool:
        return self._yolo_enabled

    @yolo_enabled.setter
    def yolo_enabled(self, value: bool):
        self._yolo_enabled = value

    @property
    def last_detection_text(self) -> str:
        return self._last_detection_text

    @property
    def last_detection_time(self) -> float:
        return self._last_detection_time

    @property
    def last_inference_text(self) -> str:
        return self._last_inference_text

    @property
    def last_inference_time(self) -> float:
        return self._last_inference_time

    @property
    def is_busy(self) -> bool:
        return self._state != STATE_IDLE or self.infer.is_busy

    # ==================== 模式控制 ====================

    def set_mode(self, mode_idx: int):
        """切换功能模式 (1-5)

        模式切换策略：
        - 模式 1（障碍物检测）：恢复 YOLO 避障线程，确保实时检测
        - 模式 2~5（文字识别/人脸/场景/问答）：暂停 YOLO 避障线程，
          释放 GPU 显存和 CUDA 核心，提高 VLM 推理效率
        """
        if 1 <= mode_idx <= 5:
            prev_mode = self._current_mode
            self._current_mode = mode_idx
            logger.info(f"切换模式: {mode_idx} - {self.mode_name}")

            # 仪表板打点
            try:
                from src.dashboard_status import system_status
                system_status.set_mode(mode_idx, self.mode_name)
                system_status.set_yolo_running(mode_idx == 1)
            except ImportError:
                pass

            # 离开模式 1 时暂停 YOLO，进入模式 1 时恢复 YOLO
            if prev_mode == 1 and mode_idx != 1:
                YOLO_PAUSE_EVENT.clear()
                logger.info("YOLO 避障已暂停（离开障碍检测模式，释放 GPU 资源）")
            elif prev_mode != 1 and mode_idx == 1:
                YOLO_PAUSE_EVENT.set()
                logger.info("YOLO 避障已恢复（进入障碍检测模式）")

            self.tts.speak(
                TIP_VOICE[self._voice_lang].get("mode_switch", "已切换至{mode}").format(mode=self.mode_name),
                priority=TTS_PRIORITY_SYSTEM
            )

    def toggle_auto(self) -> bool:
        """开关自动模式"""
        self._auto_enabled = not self._auto_enabled
        tip_key = "auto_on" if self._auto_enabled else "auto_off"
        self.tts.speak(TIP_VOICE[self._voice_lang].get(tip_key, ""), priority=TTS_PRIORITY_SYSTEM)
        logger.info(f"自动模式: {'开启' if self._auto_enabled else '关闭'}")
        return self._auto_enabled

    def set_lang(self, lang: str):
        """切换语种 (zh/en)"""
        self._voice_lang = lang
        tip_key = f"lang_switch_{lang}"
        self.tts.speak(TIP_VOICE[self._voice_lang].get(tip_key, ""), priority=TTS_PRIORITY_SYSTEM)

    # ==================== 事件回调注册 ====================

    def on(self, event: str, callback: Callable):
        """注册事件回调"""
        self._callbacks[event] = callback

    def _emit(self, event: str, *args):
        """触发事件回调"""
        cb = self._callbacks.get(event)
        if cb:
            try:
                cb(*args)
            except Exception as e:
                logger.error(f"回调异常 [{event}]: {e}")

    # ==================== 手动触发 ====================

    def handle_trigger(self, frame, force: bool = False):
        """
        处理手动触发（如空格键拍照推理）

        Args:
            frame: 当前 POV 摄像头帧
            force: 是否强制执行（忽略防抖和状态锁）
        """
        now = time.time()

        # 防抖
        if not force and (now - self._last_trigger_time < SPACE_DEBOUNCE):
            return
        self._last_trigger_time = now

        if self.is_busy and not force:
            logger.warning("系统忙，忽略触发")
            return

        # 拍照音效
        self.tts.play_shutter()

        # 异步执行推理
        threading.Thread(
            target=self._run_inference,
            args=(frame.copy(), self._current_mode),
            daemon=True
        ).start()

    # ==================== 自动模式 ====================

    def should_scan(self, now: float) -> bool:
        """是否应该执行自动扫描"""
        return (
            self._auto_enabled
            and not self.is_busy
            and (now - self._last_scan_time >= AGENT_SCAN_INTERVAL)
        )

    def auto_scan(self, frame):
        """
        自动模式场景扫描 + 任务执行

        注意：自动模式下也会调用 VLM 推理，需要暂停 YOLO 防止 GPU 竞争

        Args:
            frame: 当前 POV 摄像头帧
        """
        self._last_scan_time = time.time()

        img_b64 = self.infer.image_to_base64(frame)
        if not img_b64:
            return

        # ★ 暂停 YOLO 防止 GPU 竞争，推理完成后恢复
        yolo_was_running = YOLO_PAUSE_EVENT.is_set()
        if yolo_was_running:
            YOLO_PAUSE_EVENT.clear()
            logger.info("YOLO 避障已暂停（自动模式 VLM 推理）")

        try:
            res = self.infer.infer(AGENT_PROMPT, img_b64)
        finally:
            if yolo_was_running and self._current_mode == 1:
                YOLO_PAUSE_EVENT.set()
                logger.info("YOLO 避障已恢复（自动模式推理完成）")

        tasks = self._parse_task_numbers(res)
        logger.info(f"自动分配任务: {tasks}")

        if not tasks:
            return

        self._execute_auto_tasks(tasks, frame)

    def _execute_auto_tasks(self, tasks: list, frame):
        """执行自动模式任务（带去重静默）"""
        now = time.time()

        # 场景去重
        if tasks == self._last_tasks:
            self._same_count += 1
            if self._same_count >= 3:
                for num in tasks:
                    self._last_broadcast[num] = now + LONG_TIME_LIMIT
                self.tts.speak(TIP_VOICE[self._voice_lang].get("no_change", "画面无变化"), priority=TTS_PRIORITY_SYSTEM)
                return
        else:
            self._same_count = 0
            self._last_tasks = tasks.copy()

        # 限制最多执行 2 个任务
        run_tasks = tasks[:2]
        logger.info(f"自动执行任务（限2个）: {run_tasks}")

        for num in run_tasks:
            if now < self._last_broadcast[num] or self.infer.is_busy:
                continue
            self._last_broadcast[num] = now + BROADCAST_COOLDOWN
            threading.Thread(
                target=self._run_inference,
                args=(frame.copy(), num),
                daemon=True
            ).start()
            time.sleep(1.0)

    # ==================== YOLO 避障回调 ====================

    def on_yolo_detect(self, detection_result):
        """
        YOLO 检测结果回调（来自 detection.py）

        优先级仲裁 + 去重策略（P0 实时性优先）：
        - danger（紧急避障）：优先级 0，可打断所有低优先级播报 + 旧 P0
        - warning（普通预警）：优先级 1，可打断 VLM 播报但不打断 danger
        - P0 警报附带生成时间戳，TTS 层会丢弃过时 (>2.5s) 的警告
        - P0 去重按「方向 + 粗距离带」判定，同方向但距离显著变化会重新播报

        Args:
            detection_result: DetectionResult 实例
        """
        if not self._yolo_enabled:
            return

        msg = detection_result.alert_message
        if not msg:
            return

        now = time.time()
        has_danger = len(detection_result.dangers) > 0

        # 冷却控制：危险场景 3 秒，预警场景 8 秒
        cooldown = 3.0 if has_danger else 8.0
        if now - self._yolo_last_announce < cooldown:
            return

        # ★ P0 去重增强：方向 + 粗距离带（0.5m 粒度）
        # 同方向但距离发生显著变化（如 1.2m→0.3m）视为新危险，必须重新播报
        dedup_key = self._extract_obstacle_key(msg, has_danger, detection_result)
        last_key = getattr(self, '_yolo_last_key', '')
        if dedup_key == last_key:
            return
        self._yolo_last_key = dedup_key
        self._yolo_last_announce = now

        # 存储供 Web 仪表板读取
        self._last_detection_text = msg
        self._last_detection_time = now

        # 选择优先级：危险立即打断一切（含旧的 P0），预警只打断 VLM
        priority = TTS_PRIORITY_EMERGENCY if has_danger else TTS_PRIORITY_WARNING

        # ★ 传递检测时间戳，TTS 层据此判断过时丢弃
        detect_time = detection_result.timestamp or now
        logger.info(f"YOLO 避障播报 [P{priority}]: {msg}")
        self.tts.speak(msg, priority=priority, generated_at=detect_time)

        # 仪表板打点
        try:
            from src.dashboard_status import system_status
            system_status.log_detection(msg, priority)
        except ImportError:
            pass

    @staticmethod
    def _extract_obstacle_key(msg: str, has_danger: bool = False,
                              result=None) -> str:
        """
        从播报消息中提取去重 key

        P0（危险）：方向 + 粗距离带（0.5m 粒度），同方向但距离显著变化要重新播报
        P1（预警）：仅按方向去重，避免刷屏

        例如：
          P0 "危险！右侧近处约800毫米有人" → "右侧:800-1000"
          P0 "危险！右侧近处约300毫米有人" → "右侧:0-500"  ← 新 key，会播报
          P1 "注意，右侧有自行车" → "右侧"
        """
        # 提取方向
        m = re.search(r'(正前方|左侧|右侧|左前方|右前方)', msg)
        direction = m.group(0) if m else msg[:6]

        if not has_danger:
            return direction

        # P0：附加粗距离带以区分靠近过程
        distance_band = ""
        if result and result.dangers:
            for danger_text in result.dangers:
                dist_m = re.search(r'约(\d+)毫米', danger_text)
                if dist_m:
                    mm = int(dist_m.group(1))
                    # 粗粒度分带：0-500, 500-1000, 1000-1500
                    band = (mm // 500) * 500
                    distance_band = f":{band}-{band + 500}"
                    break

        return f"{direction}{distance_band}"

    def toggle_yolo(self) -> bool:
        """开关 YOLO 避障播报"""
        self.yolo_enabled = not self._yolo_enabled
        logger.info(f"YOLO 避障播报: {'开启' if self._yolo_enabled else '关闭'}")
        return self._yolo_enabled

    # ==================== 语音交互 ====================

    def handle_voice_chat(self, frame):
        """语音交互模式（带麦克风互斥锁 + GPU 资源保护）"""
        if self._mic_listening:
            logger.warning("麦克风正忙")
            return

        self._mic_listening = True

        # ★ 暂停 YOLO 防止 VLM 推理期间 GPU 竞争
        yolo_was_running = YOLO_PAUSE_EVENT.is_set()
        if yolo_was_running:
            YOLO_PAUSE_EVENT.clear()
            logger.info("YOLO 避障已暂停（语音交互模式）")

        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, 0.3)
                audio = r.listen(source, timeout=8)

            user_text = r.recognize_google(audio, language="zh-CN")
            logger.info(f"语音识别: {user_text}")

            # 尝试解析为任务（text-only 推理，轻量）
            task_raw = self.infer.infer(
                TASK_PLAN_PROMPT.format(user_cmd=user_text)
            )
            tasks = self._parse_task_numbers(task_raw)
            if tasks:
                self.tts.speak(TIP_VOICE[self._voice_lang].get("cmd_ok", "收到指令"), priority=TTS_PRIORITY_SYSTEM)
                self._execute_auto_tasks(tasks, frame)
                return

            # 否则作为对话（含图片的多模态推理）
            img_b64 = self.infer.image_to_base64(frame)
            history_str = ""
            for item in self._context_history:
                history_str += f"User: {item['q']}\nAssistant: {item['a']}\n"

            prompt = PROMPT_LIB[self._voice_lang][4]
            full_prompt = f"{history_str}\n{prompt}\nUser: {user_text}"
            ans = self.infer.infer(full_prompt, img_b64)

            if ans:
                self.tts.speak(ans, priority=TTS_PRIORITY_NORMAL)
                self._context_history.append({"q": user_text, "a": ans})
                if len(self._context_history) > MAX_CONTEXT_ROUND:
                    self._context_history.pop(0)

        except ImportError:
            logger.warning("speech_recognition 未安装")
        except Exception as e:
            logger.warning(f"语音识别异常: {e}")
            self.tts.speak(TIP_VOICE[self._voice_lang].get("mic_fail", "语音识别失败"), priority=TTS_PRIORITY_SYSTEM)
        finally:
            if yolo_was_running and self._current_mode == 1:
                YOLO_PAUSE_EVENT.set()
                logger.info("YOLO 避障已恢复（语音交互完成）")
            self._mic_listening = False

    # ==================== 内部方法 ====================

    def _run_inference(self, frame, mode_idx: int):
        """
        执行单次推理任务

        GPU 资源保护：触发推理时立即暂停 YOLO，不等到 set_mode 切换
        防止 YOLO 和 VLM 同时占用 GPU 导致推理耗时从 ~15s 暴涨到 120s+
        推理结束后，仅在仍处于障碍检测模式时恢复 YOLO

        Args:
            frame: 图像帧
            mode_idx: 模式编号 (1-5)
        """
        # ★ 立即暂停 YOLO：必须在任何 I/O 操作之前执行，
        # 确保 YOLO 检测线程尽快释放 GPU 显存和 CUDA 核心
        yolo_was_running = YOLO_PAUSE_EVENT.is_set()
        if yolo_was_running:
            YOLO_PAUSE_EVENT.clear()
            logger.info("YOLO 避障已暂停（VLM 推理独占 GPU）")

        # ★ 仪表板：标记识别中状态
        try:
            from src.dashboard_status import system_status
            system_status.set_app_state("inferring")
        except ImportError:
            pass

        try:
            # 保存快照
            snapshot_path = self._save_snapshot(frame)

            # 仪表板打点：记录最近快照
            try:
                from src.dashboard_status import system_status
                system_status.set_last_snapshot(snapshot_path, mode_idx)
            except ImportError:
                pass

            # 图像编码
            img_b64 = self.infer.image_to_base64(frame)
            if not img_b64:
                logger.error("图像编码失败，推理中止")
                # 编码失败也需恢复 YOLO
                if yolo_was_running and self._current_mode == 1:
                    YOLO_PAUSE_EVENT.set()
                    logger.info("YOLO 避障已恢复（编码失败）")
                return

            try:
                # 推理
                logger.info(f"推理模式 {mode_idx} ({MODE_NAMES[mode_idx - 1]})，快照: {snapshot_path}")
                prompt = PROMPT_LIB[self._voice_lang][mode_idx - 1]
                result = self.infer.infer(prompt, img_b64)
            finally:
                # 推理完成后，仅在障碍检测模式下恢复 YOLO
                if yolo_was_running and self._current_mode == 1:
                    YOLO_PAUSE_EVENT.set()
                    logger.info("YOLO 避障已恢复")
        finally:
            # ★ 仪表板：恢复空闲状态（无论推理成功/失败/异常都执行）
            try:
                from src.dashboard_status import system_status
                system_status.set_app_state("idle")
            except ImportError:
                pass

        if result:
            # 存储供 Web 仪表板读取
            self._last_inference_text = result
            self._last_inference_time = time.time()

            # 保存推理日志
            log_path = self._save_infer_log(snapshot_path, mode_idx, result)
            logger.info(f"结果: {result[:80]}...")
            logger.info(f"日志: {log_path}")

            # TTS 播报（常规优先级，可被避障打断）
            self.tts.speak(result, priority=TTS_PRIORITY_NORMAL)
        else:
            # 推理返回空（超时 / thinking 截断 / 模型异常）
            logger.warning(f"推理返回空结果 (模式{mode_idx})")
            self.tts.speak(
                TIP_VOICE[self._voice_lang].get("infer_timeout", "画面识别超时，请重试"),
                priority=TTS_PRIORITY_SYSTEM,
            )

    def _save_snapshot(self, frame) -> str:
        """保存快照图片"""
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOT_DIR, f"snap_{timestamp}.jpg")
        cv2.imwrite(path, frame)
        return path

    def _save_infer_log(self, snapshot_path: str, mode_idx: int, result: str) -> str:
        """保存推理日志"""
        os.makedirs(INFER_LOG_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(INFER_LOG_DIR, f"infer_{timestamp}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模式：{MODE_NAMES[mode_idx - 1]}\n")
            f.write(f"抓拍：{snapshot_path}\n")
            f.write("=" * 40 + "\n")
            f.write(result)
        return path

    @staticmethod
    def _parse_task_numbers(raw: str) -> list:
        """解析模型返回的任务编号"""
        try:
            nums = raw.strip().split(",")
            return [int(n.strip()) for n in nums if 1 <= int(n.strip()) <= 5]
        except Exception:
            return []

    # ==================== 生命周期 ====================

    def shutdown(self):
        """安全关闭"""
        self.tts.stop()
        self._auto_enabled = False
        self._yolo_enabled = False
        logger.info("Agent 已关闭")
