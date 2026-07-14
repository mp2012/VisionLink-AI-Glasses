"""
TTS 语音合成模块
跨平台 TTS 引擎 + 音频播报队列，防止语音重叠冲突

特性：
- Windows: PowerShell SAPI5
- Jetson: piper-tts (离线, 音质好) > espeak-ng (离线, 回退) > edge-tts (在线, 可选)
- 播报队列：新语音自动中断旧语音
- 音效播放：支持按键提示音、拍照快门音等
- 线程安全
"""
import os
import re
import shlex
import time
import logging
import subprocess
import threading
from typing import Optional

from .platform import IS_JETSON, IS_WINDOWS
from .config import (
    AUDIO_DEVICE, TTS_VOICE, TTS_ENGINE,
    TTS_PIPER_MODEL, TTS_PIPER_CONFIG, SOUND_EFFECTS,
    TTS_PRIORITY_NORMAL, TTS_PRIORITY_EMERGENCY,
    TTS_PRIORITY_SYSTEM,
    TTS_ALERT_STALE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# Markdown 格式符号匹配正则（预编译，避免每次调用重复编译）
_MD_PATTERNS = [
    (re.compile(r'\*\*(.+?)\*\*'), r'\1'),     # **加粗**
    (re.compile(r'\*(.+?)\*'),     r'\1'),     # *斜体*
    (re.compile(r'_{2,}(.+?)_{2,}'), r'\1'),   # __加粗__
    (re.compile(r'_(.+?)_'),       r'\1'),     # _斜体_
    (re.compile(r'`{1,3}(.+?)`{1,3}'), r'\1'), # `代码` 或 ```代码块```
    (re.compile(r'#{1,6}\s*'),     ''),        # 标题符号
    (re.compile(r'~~(.+?)~~'),     r'\1'),     # ~~删除线~~
    (re.compile(r'>\s*'),          ''),        # 引用 > 符号
    (re.compile(r'[-*]\s{1,2}'),   ''),        # 列表符号 - 或 *
]


def strip_markdown_for_tts(text: str) -> str:
    """
    去除 Markdown 格式符号，避免 TTS 朗读出星号等噪音字符。

    例: "**特写镜头**" → "特写镜头"
        "# 标题" → "标题"
        "`code` and **bold**" → "code and bold"

    只在边缘添加空格分隔，不破坏原始语义。
    """
    for pattern, replacement in _MD_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


class TTSEngine:
    """
    跨平台 TTS 引擎，异步非阻塞
    新语音自动中断旧语音，防止播报重叠

    Usage:
        tts = TTSEngine()
        tts.speak("前方有障碍物")
        tts.play_shutter()       # 播放快门音效
        tts.stop()               # 停止当前语音
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._muted = False
        self._tts_method = None  # 延迟检测
        self._current_priority: Optional[int] = None  # 当前播放内容的优先级

    def _detect_tts_method(self) -> str:
        """检测可用的 TTS 方法（按优先级）"""
        if not IS_JETSON:
            return "edge"

        engine = TTS_ENGINE

        if engine == "piper":
            if (TTS_PIPER_MODEL and os.path.exists(TTS_PIPER_MODEL)
                    and self._check_cmd("piper --help")):
                logger.info("TTS: piper-tts (离线)")
                return "piper"
            else:
                logger.warning("piper 模型或程序不可用，回退到 espeak-ng")

        if engine in ("piper", "espeak"):
            if self._check_cmd("espeak-ng --help"):
                logger.info("TTS: espeak-ng (离线)")
                return "espeak"
            else:
                logger.warning("espeak-ng 不可用，回退到 edge-tts")

        # edge-tts 在线模式
        logger.info("TTS: edge-tts (在线)")
        return "edge"

    @staticmethod
    def _check_cmd(cmd: str) -> bool:
        """检查命令是否可用"""
        try:
            subprocess.run(
                cmd.split(), capture_output=True, timeout=3,
                stdin=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    # ==================== 主控接口 ====================

    def speak(self, text: str, priority: int = None, generated_at: float = None):
        """
        异步播放语音，自动中断旧语音（遵循优先级规则）

        Args:
            text: 播报文本
            priority: 优先级（数字越小越高），默认 TTS_PRIORITY_NORMAL(3)
                      0=紧急避障, 1=普通预警, 2=系统提示, 3=常规播报
                      特殊：同优先级 P0 允许互相打断（新危险替换旧危险）
            generated_at: 警报生成时间戳（time.time()），用于过时检查
                          仅 P0 警报需要传入，用于丢弃过时信息
        """
        if priority is None:
            priority = TTS_PRIORITY_NORMAL

        text = text.strip()
        if not text or self._muted:
            return

        if self._tts_method is None:
            self._tts_method = self._detect_tts_method()

        def _worker():
            # ★ 过时检查：P0 危险警报超过阈值直接丢弃，不播过时信息
            if priority == TTS_PRIORITY_EMERGENCY and generated_at is not None:
                age = time.time() - generated_at
                if age > TTS_ALERT_STALE_THRESHOLD:
                    logger.warning(
                        f"P0 警报已过时 ({age:.1f}s > {TTS_ALERT_STALE_THRESHOLD}s)，丢弃不播: "
                        f"{text[:60]}..."
                    )
                    return

            # Markdown 清洗：去除格式符号，避免 TTS 朗读出 "星号星号 XX 星号星号" 等噪音
            clean_text = strip_markdown_for_tts(text)
            clean_text = clean_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            clean_text = " ".join(clean_text.split())

            # 优先级仲裁：
            # - 高优先级可打断低优先级 (new < current)
            # - 同优先级状态通知允许替换（P0 紧急/P1 预警/P2 系统通知）
            #   只有 P3（场景描述/问答结果）的完整内容不能被同优先级覆盖
            # - 过时的 P0 警报在入口处已丢弃，此处为二次保障
            with self._lock:
                if self._process is not None and self._process.poll() is None:
                    if self._current_priority is not None:
                        # 允许打断的情况：
                        #   1. 严格更高优先级：new < current
                        #   2. 同优先级且属于可替换状态通知 (P0/P1/P2)
                        #      P3 不在此列，避免打断正在播放的场景描述
                        can_interrupt = (
                            priority < self._current_priority
                            or (priority == self._current_priority
                                and priority <= TTS_PRIORITY_SYSTEM)
                        )
                        if not can_interrupt:
                            logger.debug(
                                f"语音被优先级仲裁拒绝 "
                                f"(新=P{priority} 无法打断 当前=P{self._current_priority}): "
                                f"{clean_text[:40]}..."
                            )
                            return

            # 停止旧语音并立即播放新语音
            self.stop()
            with self._lock:
                self._current_priority = priority

            # 确定实际使用的引擎（piper 遇到英文会切 espeak）
            actual_method = self._tts_method
            if IS_JETSON and actual_method == "piper" and self._is_mostly_english(clean_text):
                actual_method = "espeak(en)"
            logger.info(f"播放语音 [P{priority}][{actual_method}]: {clean_text[:60]}...")

            # 仪表板打点：记录 TTS 开始播放
            try:
                from src.dashboard_status import system_status
                system_status.log_tts(priority, clean_text, actual_method)
            except ImportError:
                pass

            try:
                with self._lock:
                    if IS_JETSON:
                        self._process = self._speak_jetson(clean_text)
                    elif IS_WINDOWS:
                        self._process = self._speak_windows(clean_text)
                    else:
                        self._process = self._speak_jetson(clean_text)

                    # 等待播放完成
                    if self._process:
                        self._process.wait()
            except Exception as e:
                logger.error(f"语音播放异常: {e}")
                self._process = None
            finally:
                # 播放完成后清除优先级（仅当没有被新的高优先级播报覆盖时）
                with self._lock:
                    if self._current_priority == priority:
                        self._current_priority = None
                    self._process = None

                # 仪表板打点：TTS 空闲
                try:
                    from src.dashboard_status import system_status
                    system_status.set_tts_idle()
                except ImportError:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    def stop(self):
        """强制终止当前语音（包括杀掉占用音频设备的 aplay 进程）"""
        with self._lock:
            if self._process is not None:
                try:
                    # 先杀 shell 进程组（包括管道中的所有子进程）
                    pid = self._process.pid
                    if pid:
                        try:
                            # kill 整个进程组，确保 aplay 也被杀掉
                            os.killpg(os.getpgid(pid), 2)  # SIGINT
                            time.sleep(0.1)
                            os.killpg(os.getpgid(pid), 9)  # SIGKILL 兜底
                        except (ProcessLookupError, OSError):
                            pass
                    self._process.wait(timeout=1.0)
                    logger.debug("已终止上一条语音")
                except Exception as e:
                    logger.warning(f"终止语音异常: {e}")
                    # 不执行系统级 pkill，避免误杀其他应用的 aplay 进程
                self._process = None

    def mute(self):
        """静音模式"""
        self.stop()
        self._muted = True

    def unmute(self):
        """取消静音"""
        self._muted = False

    def is_speaking(self) -> bool:
        """是否正在播放语音"""
        with self._lock:
            if self._process is None:
                return False
            return self._process.poll() is None

    @property
    def speaking_priority(self) -> Optional[int]:
        """当前播放内容的优先级（None=空闲）"""
        return self._current_priority

    # ==================== 音效接口 ====================

    def play_effect(self, filepath: str):
        """播放指定音效文件（异步）"""
        if not os.path.exists(filepath):
            logger.warning(f"音效文件不存在: {filepath}")
            return
        threading.Thread(
            target=lambda: self._play_audio_file(filepath),
            daemon=True
        ).start()

    def play_shutter(self):
        """播放拍照快门音效"""
        shutter_path = SOUND_EFFECTS.get("shutter")
        if shutter_path and os.path.exists(shutter_path):
            self.play_effect(shutter_path)
        else:
            self.play_beep()

    def play_beep(self):
        """播放按键提示音"""
        threading.Thread(target=self._play_beep_impl, daemon=True).start()

    def play_alert(self):
        """播放告警提示音"""
        threading.Thread(target=self._play_alert_impl, daemon=True).start()

    # ==================== 平台实现 ====================

    @staticmethod
    def _play_audio_file(filepath: str):
        """通用音频文件播放，ALSA 直连确保双声道"""
        audio_device = AUDIO_DEVICE or "plughw:1,0"
        os.system(
            f"ffmpeg -i {shlex.quote(filepath)} -af 'pan=stereo|c0=c0|c1=c0' -ar 48000 "
            f"-f wav - 2>/dev/null | "
            f"aplay -D {shlex.quote(audio_device)} -q 2>/dev/null"
        )

    def _play_beep_impl(self):
        """播放短促按键提示音"""
        if IS_WINDOWS:
            import winsound
            winsound.Beep(800, 80)
        elif IS_JETSON:
            beep_path = "/tmp/beep.wav"
            if not os.path.exists(beep_path):
                os.system(
                    f"sox -n -r 22050 -c 1 {shlex.quote(beep_path)} synth 0.05 sine 800 vol 0.3 2>/dev/null"
                    f" || ffmpeg -f lavfi -i 'sine=frequency=800:duration=0.05' "
                    f"-ar 22050 -ac 1 {shlex.quote(beep_path)} -y 2>/dev/null"
                )
            os.system(f"aplay -D {shlex.quote(AUDIO_DEVICE)} -q {shlex.quote(beep_path)} 2>/dev/null")

    def _play_alert_impl(self):
        """播放告警提示音（比普通 beep 更显著）"""
        if IS_WINDOWS:
            import winsound
            winsound.Beep(1200, 200)
        elif IS_JETSON:
            alert_path = "/tmp/alert.wav"
            if not os.path.exists(alert_path):
                os.system(
                    f"ffmpeg -f lavfi -i 'sine=frequency=1200:duration=0.2' "
                    f"-ar 22050 -ac 1 {shlex.quote(alert_path)} -y 2>/dev/null"
                )
            os.system(f"aplay -D {shlex.quote(AUDIO_DEVICE)} -q {shlex.quote(alert_path)} 2>/dev/null")

    @staticmethod
    def _speak_windows(text: str):
        """Windows PowerShell SAPI5 TTS"""
        safe_text = text.replace("'", "''")
        cmd = (
            f'powershell "Add-Type -AssemblyName System.Speech;'
            f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            f'$synth.Rate = -2;'
            f'$synth.Speak(\'{safe_text}\')"'
        )
        return subprocess.Popen(
            cmd, shell=True,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )

    # ==================== Jetson TTS 实现 ====================

    @staticmethod
    def _is_mostly_english(text: str) -> bool:
        """
        判断文本是否主要为英文
        当英文/数字/符号占比超过 60% 时，认为需要英文 TTS 引擎
        """
        if not text:
            return False
        en_count = sum(1 for c in text if c.isascii() and c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return False
        return (en_count / total_alpha) > 0.6

    def _speak_jetson(self, text: str):
        """Jetson TTS：piper > espeak-ng > edge-tts 自动选择"""
        method = self._tts_method or "piper"

        # piper 中文模型朗读英文会乱说，自动切换到 espeak（英文支持好）
        if method == "piper" and self._is_mostly_english(text):
            if self._check_cmd("espeak-ng --help"):
                logger.debug("检测到英文文本，自动切换 espeak-ng 播报")
                return self._speak_espeak_english(text)

        if method == "piper":
            return self._speak_piper(text)
        elif method == "espeak":
            return self._speak_espeak(text)
        else:
            return self._speak_edge(text)

    @staticmethod
    def _speak_piper(text: str) -> Optional[subprocess.Popen]:
        """
        Piper TTS（离线，音质好）
        流程：piper 合成 raw PCM → ffmpeg 转 wav → aplay 直连 ALSA 播放
        """
        tmp_wav = f"/tmp/tts_{time.time_ns()}.wav"
        safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        audio_device = AUDIO_DEVICE or "plughw:1,0"
        cmd = (
            f'echo "{safe_text}" | '
            f"piper -m {shlex.quote(TTS_PIPER_MODEL)} --config {shlex.quote(TTS_PIPER_CONFIG)} "
            f"--output-raw 2>/dev/null | "
            f"ffmpeg -f s16le -ar 22050 -ac 1 -i - "
            f"-af 'pan=stereo|c0=c0|c1=c0' -ar 48000 "
            f"-f wav {shlex.quote(tmp_wav)} -y 2>/dev/null && "
            f"aplay -D {shlex.quote(audio_device)} -q {shlex.quote(tmp_wav)} 2>/dev/null; "
            f"rm -f {shlex.quote(tmp_wav)}"
        )
        return subprocess.Popen(
            cmd, shell=True, executable="/bin/bash",
            preexec_fn=os.setsid  # 创建新进程组，方便 killpg
        )

    @staticmethod
    def _speak_espeak(text: str) -> Optional[subprocess.Popen]:
        """
        eSpeak-NG TTS（离线，轻量回退）- 中文
        """
        tmp_wav = f"/tmp/tts_{time.time_ns()}.wav"
        safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        audio_device = AUDIO_DEVICE or "plughw:1,0"
        cmd = (
            f'espeak-ng -v zh-cmn "{safe_text}" -w {shlex.quote(tmp_wav)} -s 160 2>/dev/null && '
            f"aplay -D {shlex.quote(audio_device)} -q {shlex.quote(tmp_wav)} 2>/dev/null; "
            f"rm -f {shlex.quote(tmp_wav)}"
        )
        return subprocess.Popen(
            cmd, shell=True, executable="/bin/bash",
            preexec_fn=os.setsid
        )

    @staticmethod
    def _speak_espeak_english(text: str) -> Optional[subprocess.Popen]:
        """
        eSpeak-NG TTS - 英文模式
        piper 中文模型遇到英文会乱说，自动切换到此方法
        """
        tmp_wav = f"/tmp/tts_{time.time_ns()}.wav"
        safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        audio_device = AUDIO_DEVICE or "plughw:1,0"
        cmd = (
            f'espeak-ng -v en-us "{safe_text}" -w {shlex.quote(tmp_wav)} -s 150 2>/dev/null && '
            f"aplay -D {shlex.quote(audio_device)} -q {shlex.quote(tmp_wav)} 2>/dev/null; "
            f"rm -f {shlex.quote(tmp_wav)}"
        )
        return subprocess.Popen(
            cmd, shell=True, executable="/bin/bash",
            preexec_fn=os.setsid
        )

    @staticmethod
    def _speak_edge(text: str) -> Optional[subprocess.Popen]:
        """
        Edge-TTS（在线，微软神经网络语音）
        需要网络连接
        """
        safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        tmp_path = f"/tmp/tts_{time.time_ns()}.mp3"
        voice = TTS_VOICE or "zh-CN-XiaoxiaoNeural"
        audio_device = AUDIO_DEVICE or "plughw:1,0"
        cmd = (
            f'edge-tts --voice {shlex.quote(voice)} --text "{safe_text}" '
            f"--write-media {shlex.quote(tmp_path)} 2>/dev/null && "
            f"aplay -D {shlex.quote(audio_device)} -q {shlex.quote(tmp_path)} 2>/dev/null; "
            f"rm -f {shlex.quote(tmp_path)}"
        )
        return subprocess.Popen(
            cmd, shell=True, executable="/bin/bash",
            preexec_fn=os.setsid
        )
