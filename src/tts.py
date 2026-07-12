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
import time
import logging
import subprocess
import threading
from typing import Optional

from .platform import IS_JETSON, IS_WINDOWS
from .config import (
    AUDIO_DEVICE, TTS_VOICE, TTS_ENGINE,
    TTS_PIPER_MODEL, TTS_PIPER_CONFIG, SOUND_EFFECTS,
)

logger = logging.getLogger(__name__)


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

    def speak(self, text: str):
        """异步播放语音，自动中断旧语音"""
        text = text.strip()
        if not text or self._muted:
            return

        if self._tts_method is None:
            self._tts_method = self._detect_tts_method()

        def _worker():
            self.stop()
            clean_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            clean_text = " ".join(clean_text.split())

            # 确定实际使用的引擎（piper 遇到英文会切 espeak）
            actual_method = self._tts_method
            if IS_JETSON and actual_method == "piper" and self._is_mostly_english(clean_text):
                actual_method = "espeak(en)"
            logger.info(f"播放语音 [{actual_method}]: {clean_text[:60]}...")

            try:
                with self._lock:
                    if IS_JETSON:
                        self._process = self._speak_jetson(clean_text)
                    elif IS_WINDOWS:
                        self._process = self._speak_windows(clean_text)
                    else:
                        self._process = self._speak_jetson(clean_text)
            except Exception as e:
                logger.error(f"语音播放异常: {e}")
                self._process = None

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
                    # 暴力兜底：直接杀所有 aplay 进程
                    try:
                        os.system("pkill -9 aplay 2>/dev/null")
                    except Exception:
                        pass
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
            f"ffmpeg -i {filepath} -af 'pan=stereo|c0=c0|c1=c0' -ar 48000 "
            f"-f wav - 2>/dev/null | "
            f"aplay -D {audio_device} -q 2>/dev/null"
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
                    f"sox -n -r 22050 -c 1 {beep_path} synth 0.05 sine 800 vol 0.3 2>/dev/null"
                    f" || ffmpeg -f lavfi -i 'sine=frequency=800:duration=0.05' "
                    f"-ar 22050 -ac 1 {beep_path} -y 2>/dev/null"
                )
            os.system(f"aplay -D {AUDIO_DEVICE} -q {beep_path} 2>/dev/null")

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
                    f"-ar 22050 -ac 1 {alert_path} -y 2>/dev/null"
                )
            os.system(f"aplay -D {AUDIO_DEVICE} -q {alert_path} 2>/dev/null")

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
            f"piper -m {TTS_PIPER_MODEL} --config {TTS_PIPER_CONFIG} "
            f"--output-raw 2>/dev/null | "
            f"ffmpeg -f s16le -ar 22050 -ac 1 -i - "
            f"-af 'pan=stereo|c0=c0|c1=c0' -ar 48000 "
            f"-f wav {tmp_wav} -y 2>/dev/null && "
            f"aplay -D {audio_device} -q {tmp_wav} 2>/dev/null; "
            f"rm -f {tmp_wav}"
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
            f'espeak-ng -v zh-cmn "{safe_text}" -w {tmp_wav} -s 160 2>/dev/null && '
            f"aplay -D {audio_device} -q {tmp_wav} 2>/dev/null; "
            f"rm -f {tmp_wav}"
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
            f'espeak-ng -v en-us "{safe_text}" -w {tmp_wav} -s 150 2>/dev/null && '
            f"aplay -D {audio_device} -q {tmp_wav} 2>/dev/null; "
            f"rm -f {tmp_wav}"
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
            f'edge-tts --voice {voice} --text "{safe_text}" '
            f"--write-media {tmp_path} 2>/dev/null && "
            f"aplay -D {audio_device} -q {tmp_path} 2>/dev/null; "
            f"rm -f {tmp_path}"
        )
        return subprocess.Popen(cmd, shell=True)
