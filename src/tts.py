"""
TTS 语音合成模块
Windows: PowerShell SAPI5（异步线程，可中断）
Jetson: edge-tts + ffplay（微软神经网络语音，中文自然）
统一接口：speak(text) / stop()
"""
import os
import time
import logging
import subprocess
import threading

from .platform import IS_JETSON, IS_WINDOWS

logger = logging.getLogger(__name__)

# Jetson 音频设备: card 0 = USB Audio (AB13X, 耳麦/摄像头音频)
JETSON_AUDIO_DEV = "plughw:0,0"


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
        """使用 edge-tts 微软神经网络语音，中文自然流畅"""
        clean_text = text.replace("'", " ")
        # 生成临时 mp3 然后播放
        tmp_path = f"/tmp/tts_{time.time_ns()}.mp3"
        cmd = (
            f"edge-tts --voice zh-CN-XiaoxiaoNeural --text '{clean_text}' "
            f"--write-media {tmp_path} 2>/dev/null && "
            f"ffplay -nodisp -autoexit -loglevel quiet {tmp_path} 2>/dev/null; "
            f"rm -f {tmp_path}"
        )
        return subprocess.Popen(cmd, shell=True)

    def play_effect(self, filepath: str):
        """播放指定音效文件（异步）"""
        if not os.path.exists(filepath):
            logger.warning(f"音效文件不存在: {filepath}")
            return
        if IS_JETSON:
            threading.Thread(
                target=lambda: os.system(
                    f"ffplay -nodisp -autoexit -loglevel quiet {filepath} 2>/dev/null"
                ),
                daemon=True
            ).start()
        elif IS_WINDOWS:
            threading.Thread(
                target=lambda: os.system(f'start /min wmplayer "{filepath}" 2>nul'),
                daemon=True
            ).start()

    def play_beep(self):
        """播放按键提示音"""
        if IS_WINDOWS:
            import winsound
            winsound.Beep(800, 80)
        elif IS_JETSON:
            # 生成短促提示音：800Hz 正弦波，0.05秒
            beep_path = "/tmp/beep.wav"
            if not os.path.exists(beep_path):
                os.system(
                    f"sox -n -r 22050 -c 1 {beep_path} synth 0.05 sine 800 vol 0.3 2>/dev/null"
                    f" || ffmpeg -f lavfi -i 'sine=frequency=800:duration=0.05' -ar 22050 -ac 1 {beep_path} -y 2>/dev/null"
                )
            threading.Thread(
                target=lambda: os.system(f"aplay -D {JETSON_AUDIO_DEV} -q {beep_path} 2>/dev/null"),
                daemon=True
            ).start()
