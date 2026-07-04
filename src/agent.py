"""
自动模式 Agent
场景检测 → 任务分配 → 去重静默 → 异步执行
"""
import time
import logging
import threading

from .config import (
    AGENT_SCAN_INTERVAL, BROADCAST_COOLDOWN, LONG_TIME_LIMIT,
    MAX_CONTEXT_ROUND, STATE_IDLE, STATE_LISTEN,
)
from .prompts import AGENT_PROMPT, TASK_PLAN_PROMPT, TIP_VOICE, PROMPT_LIB

logger = logging.getLogger(__name__)


class AutoAgent:
    """自动模式任务调度器"""

    def __init__(self, inference_engine, tts_engine):
        self.infer = inference_engine
        self.tts = tts_engine
        self._enabled = False
        self._last_run_time = 0
        self._last_broadcast = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self._last_tasks = []
        self._same_count = 0
        self._context_history = []
        self._mic_listening = False
        self._voice_lang = "zh"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self):
        self._enabled = not self._enabled
        return self._enabled

    def set_lang(self, lang: str):
        self._voice_lang = lang

    def should_scan(self, now: float) -> bool:
        return self._enabled and (now - self._last_run_time >= AGENT_SCAN_INTERVAL) and not self.infer.is_busy

    def detect_tasks(self, frame) -> list:
        """分析场景，返回任务编号列表"""
        from .inference import InferenceEngine
        img_b64 = InferenceEngine.image_to_base64(frame)
        if not img_b64:
            return []
        res = self.infer.infer(AGENT_PROMPT, img_b64)
        tasks = self._parse_task_numbers(res)
        self._last_run_time = time.time()
        logger.info(f"自动分配任务：{tasks}")
        return tasks

    def execute_tasks(self, tasks: list, frame, func_map: dict):
        """执行任务列表（带去重静默逻辑）"""
        now = time.time()

        # 场景去重
        if tasks == self._last_tasks and len(tasks) > 0:
            self._same_count += 1
            if self._same_count >= 3:
                for num in tasks:
                    self._last_broadcast[num] = now + LONG_TIME_LIMIT
                self.tts.speak(TIP_VOICE[self._voice_lang]["no_change"])
                return
        else:
            self._same_count = 0
            self._last_tasks = tasks.copy()

        run_tasks = tasks[:2]
        logger.info(f"自动模式执行任务（限2个）：{run_tasks}")
        for num in run_tasks:
            if now < self._last_broadcast[num] or self.infer.is_busy:
                continue
            func = func_map.get(num)
            if func:
                threading.Thread(target=func, args=(frame.copy(),), daemon=True).start()
                self._last_broadcast[num] = now + BROADCAST_COOLDOWN
                time.sleep(1.0)

    def handle_voice_chat(self, frame):
        """语音交互模式（带麦克风互斥锁）"""
        if self._mic_listening:
            logger.warning("麦克风正忙，忽略")
            return

        self._mic_listening = True
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, 0.3)
                audio = r.listen(source, timeout=8)

            user_text = r.recognize_google(audio, language="zh-CN")
            logger.info(f"语音识别：{user_text}")

            # 尝试解析为任务
            task_raw = self.infer.infer(TASK_PLAN_PROMPT.format(user_cmd=user_text))
            tasks = self._parse_task_numbers(task_raw)
            if tasks:
                self.tts.speak(TIP_VOICE[self._voice_lang]["cmd_ok"])
                self.execute_tasks(tasks, frame, {})
                return

            # 否则作为对话
            img_b64 = self.infer.image_to_base64(frame)
            history_str = ""
            for item in self._context_history:
                history_str += f"User: {item['q']}\nAssistant: {item['a']}\n"
            full_prompt = f"{history_str}\n{PROMPT_LIB[self._voice_lang][4]}\nUser: {user_text}"
            ans = self.infer.infer(full_prompt, img_b64)
            if ans:
                self.tts.speak(ans)
                self._context_history.append({"q": user_text, "a": ans})
                if len(self._context_history) > MAX_CONTEXT_ROUND:
                    self._context_history.pop(0)

        except Exception as e:
            logger.warning(f"语音识别异常：{e}")
            self.tts.speak(TIP_VOICE[self._voice_lang]["mic_fail"])
        finally:
            self._mic_listening = False

    @staticmethod
    def _parse_task_numbers(raw: str) -> list:
        """解析模型返回的任务编号"""
        try:
            nums = raw.strip().split(",")
            return [int(n.strip()) for n in nums if 1 <= int(n.strip()) <= 5]
        except Exception:
            return []
