"""
TTS 语音合成模块
Windows: PowerShell SAPI5（异步线程，可中断）
Jetson: espeak + aplay（异步线程，可中断）
统一接口：speak(text) / stop()
"""
import os
import time
import logging
import subprocess
import threading

from .platform import IS_JETSON, IS_WINDOWS

logger = logging.getLogger(__name__)


class TTSEngine:
    """跨平台 TTS 引擎，异步非阻塞"""

    def __init__(self):
        self._process = None
        self._lock = threading.Lock()

    def stop(self):
        """强制终止当前语音"""
        with self._lock:
            if self._process is not None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=1.0)
                    logger.info("已终止上一条语音")
                except Exception as e:
                    logger.warning(f"终止语音异常: {e}")
                self._process = None

    def speak(self, text: str):
        """异步播放语音，自动中断旧语音"""
        text = text.strip()
        if not text:
            return

        def _worker():
            self.stop()
            clean_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            clean_text = " ".join(clean_text.split())
            logger.info(f"播放语音: {clean_text[:60]}...")

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

    @staticmethod
    def _speak_windows(text: str):
        safe_text = text.replace("'", "''")
        cmd = (
            f'powershell "Add-Type -AssemblyName System.Speech;'
            f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            f'$synth.Rate = -2;'
            f'$synth.Speak(\'{safe_text}\')"'
        )
        return subprocess.Popen(cmd, shell=True)

    @staticmethod
    def _speak_jetson(text: str):
        clean_text = text.replace("'", " ")
        timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
        wav_path = f"audio_cache/voice_{timestamp}.wav"
        os.makedirs("audio_cache", exist_ok=True)

        cmd = (
            f"espeak -v zh+f2 -s 140 --wav={wav_path} '{clean_text}' >/dev/null 2>&1;"
            f"aplay -q {wav_path} >/dev/null 2>&1"
        )
        return subprocess.Popen(cmd, shell=True)

    def play_beep(self):
        """播放按键提示音"""
        if IS_WINDOWS:
            import winsound
            winsound.Beep(800, 80)
        elif IS_JETSON:
            os.system("speaker-test -t sine -f 900 -l 0.06 >/dev/null 2>&1")
