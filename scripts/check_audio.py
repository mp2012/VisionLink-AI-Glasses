#!/usr/bin/env python3
"""
音频诊断脚本 - 全面测试音频设备
1. 检测音频硬件
2. 测试 beep 音效
3. 测试 TTS 语音合成
4. 测试音效文件播放
"""
import sys
import os
import time
import subprocess
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from src.platform import IS_JETSON, IS_WINDOWS
except ImportError:
    IS_JETSON = os.uname().machine.startswith("aarch64") if hasattr(os, 'uname') else False
    IS_WINDOWS = sys.platform == "win32"

print("=" * 60)
print(f"  音频诊断工具")
print(f"  平台: {'Jetson' if IS_JETSON else 'Windows' if IS_WINDOWS else 'Linux x86'}")
print("=" * 60)

errors = []
warnings = []

# ========== 1. 检测音频硬件设备 ==========
print("\n[1] 音频硬件设备检测:")

if not IS_WINDOWS:
    # aplay -l 列出声卡
    result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
    print(f"  aplay -l:")
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            print(f"    {line}")
    else:
        print(f"    ✗ 无法获取: {result.stderr.strip()}")
        errors.append("aplay 无法获取声卡列表")

    # aplay -L 列出 PCM 设备
    result2 = subprocess.run(["aplay", "-L"], capture_output=True, text=True, timeout=5)
    print(f"\n  aplay -L (PCM 设备):")
    if result2.returncode == 0:
        for line in result2.stdout.strip().split('\n'):
            if line.strip():
                print(f"    {line}")
    else:
        errors.append("aplay 无法获取 PCM 设备列表")

    # 检查 pulseaudio
    result3 = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=5)
    if result3.returncode == 0:
        print(f"\n  PulseAudio: ✓ 运行中")
    else:
        warnings.append("PulseAudio 未运行，某些音频功能可能受限")

    # 检查音频设备是否被占用
    result4 = subprocess.run(["fuser", "/dev/snd/*"], capture_output=True, text=True, timeout=3)
    if result4.stdout.strip():
        print(f"\n  ⚠ 音频设备被以下进程占用:")
        print(f"    {result4.stdout.strip()}")
        warnings.append("音频设备被其他进程占用")

# ========== 2. 测试 beep 音效 ==========
print("\n[2] 测试 beep 提示音:")
print("    (如果听到 '嘀' 一声短音，说明喇叭正常)")

from src.config import AUDIO_DEVICE

if IS_JETSON:
    audio_dev = AUDIO_DEVICE or "plughw:0,0"
    print(f"  使用设备: {audio_dev}")

    # 方法1: aplay 播放 WAV
    beep_wav = "/tmp/test_beep.wav"
    # 生成 beep wav
    gen_result = subprocess.run(
        ["sox", "-n", "-r", "22050", "-c", "1", beep_wav,
         "synth", "0.1", "sine", "800", "vol", "0.5"],
        capture_output=True, text=True, timeout=5
    )
    if gen_result.returncode == 0:
        print("  ✓ WAV 生成成功，正在播放...")
        play_result = subprocess.run(
            ["aplay", "-D", audio_dev, beep_wav],
            capture_output=True, text=True, timeout=5
        )
        if play_result.returncode == 0:
            print("  ✓ beep 播放成功")
        else:
            print(f"  ✗ beep 播放失败: {play_result.stderr.strip()}")
            errors.append(f"aplay 播放失败: {play_result.stderr.strip()}")

            # 尝试 ffplay 备选
            print("\n  尝试 ffplay 备选方案...")
            gen2 = subprocess.run(
                ["ffmpeg", "-f", "lavfi", "-i", "sine=frequency=800:duration=0.1",
                 "-ar", "22050", "-ac", "1", "/tmp/test_beep_ff.wav", "-y"],
                capture_output=True, text=True, timeout=5
            )
            if gen2.returncode == 0:
                play2 = subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "/tmp/test_beep_ff.wav"],
                    capture_output=True, text=True, timeout=5
                )
                if play2.returncode == 0:
                    print("  ✓ ffplay beep 播放成功")
                else:
                    print(f"  ✗ ffplay 播放失败: {play2.stderr.strip()}")
                    errors.append("ffplay 播放也失败")
    else:
        print(f"  ✗ WAV 生成失败: {gen_result.stderr.strip()}")
        errors.append("sox 生成 beep 失败")

elif IS_WINDOWS:
    try:
        import winsound
        winsound.Beep(800, 100)
        print("  ✓ Windows beep 播放成功")
    except Exception as e:
        print(f"  ✗ beep 失败: {e}")
        errors.append(f"Windows beep 失败: {e}")

# ========== 3. 测试音效文件播放 ==========
print("\n[3] 测试音效文件播放:")
shutter_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "audio", "pic.mp3"
)

if os.path.exists(shutter_path):
    print(f"  ✓ 快门音效文件存在: {shutter_path}")
    if IS_JETSON:
        # 用 ffplay 测试播放
        print("  正在播放快门音效（约 0.3 秒）...")
        result = subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", shutter_path],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("  ✓ 音效文件播放成功")
        else:
            print(f"  ✗ 音效播放失败: {result.stderr.strip()}")
            errors.append(f"ffplay 播放 {shutter_path} 失败")
    else:
        print("  (非 Jetson 平台，跳过播放测试)")
else:
    print(f"  ✗ 快门音效文件不存在: {shutter_path}")
    warnings.append(f"音效文件缺失: {shutter_path}")

# ========== 4. 测试 edge-tts 语音合成 ==========
print("\n[4] 测试 TTS 语音合成 (edge-tts):")
print("    (如果听到 '这是一段测试语音' 则 TTS 正常)")

# 检查 edge-tts 是否安装
result = subprocess.run(["which", "edge-tts"], capture_output=True, text=True, timeout=3)
if result.returncode != 0:
    result = subprocess.run(["pip3", "show", "edge-tts"], capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        print("  ✗ edge-tts 未安装")
        errors.append("edge-tts 未安装")
    else:
        print("  ✓ edge-tts 已安装 (pip)")

        # 尝试用 python -m 方式调用
        test_text = "这是一段测试语音"
        tmp_mp3 = "/tmp/test_tts.mp3"
        print(f"  正在合成语音: '{test_text}'")

        gen_result = subprocess.run(
            ["python3", "-m", "edge_tts", "--voice", "zh-CN-XiaoxiaoNeural",
             "--text", test_text, "--write-media", tmp_mp3],
            capture_output=True, text=True, timeout=30
        )
        if gen_result.returncode == 0 and os.path.exists(tmp_mp3):
            print(f"  ✓ TTS 合成成功 ({os.path.getsize(tmp_mp3)} bytes)")
            print("  正在播放...")
            play_result = subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_mp3],
                capture_output=True, text=True, timeout=10
            )
            if play_result.returncode == 0:
                print("  ✓ TTS 播放成功")
            else:
                print(f"  ✗ TTS 播放失败: {play_result.stderr.strip()}")
                errors.append("ffplay TTS 播放失败")
            os.remove(tmp_mp3)
        else:
            print(f"  ✗ TTS 合成失败")
            if gen_result.stderr:
                print(f"    错误: {gen_result.stderr.strip()[:200]}")
            errors.append("edge-tts 合成失败")
else:
    print("  ✓ edge-tts 已安装 (PATH)")

    test_text = "这是一段测试语音"
    tmp_mp3 = "/tmp/test_tts.mp3"
    print(f"  正在合成语音: '{test_text}'")

    gen_result = subprocess.run(
        ["edge-tts", "--voice", "zh-CN-XiaoxiaoNeural",
         "--text", test_text, "--write-media", tmp_mp3],
        capture_output=True, text=True, timeout=30
    )
    if gen_result.returncode == 0 and os.path.exists(tmp_mp3):
        print(f"  ✓ TTS 合成成功 ({os.path.getsize(tmp_mp3)} bytes)")
        print("  正在播放...")
        play_result = subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_mp3],
            capture_output=True, text=True, timeout=10
        )
        if play_result.returncode == 0:
            print("  ✓ TTS 播放成功")
        else:
            print(f"  ✗ TTS 播放失败: {play_result.stderr.strip()}")
            errors.append("ffplay TTS 播放失败")
        os.remove(tmp_mp3)
    else:
        print(f"  ✗ TTS 合成失败")
        if gen_result.stderr:
            print(f"    错误: {gen_result.stderr.strip()[:200]}")
        errors.append("edge-tts 合成失败")

# ========== 5. 测试项目 TTSEngine ==========
print("\n[5] 测试项目 TTSEngine 模块:")
try:
    from src.tts import TTSEngine

    tts = TTSEngine()
    print("  ✓ TTSEngine 初始化成功")

    # 测试 beep
    print("  播放 beep 提示音...")
    tts.play_beep()
    time.sleep(0.5)
    print("  ✓ beep 播放完成")

    # 测试快门音效
    print("  播放快门音效...")
    tts.play_shutter()
    time.sleep(0.8)
    print("  ✓ 快门音效播放完成")

    # 测试 TTS 语音
    print("  播放 TTS 语音: '音频测试成功'")
    tts.speak("音频测试成功")
    time.sleep(3.0)  # 等待播放完成
    if tts.is_speaking():
        print("  ⚠ TTS 仍在播放中（可能卡住）")
        tts.stop()
    else:
        print("  ✓ TTS 播放完成")

except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    errors.append(f"TTSEngine 导入失败: {e}")
except Exception as e:
    print(f"  ✗ 测试异常: {e}")
    errors.append(f"TTSEngine 测试异常: {e}")

# ========== 6. 总结 ==========
print("\n" + "=" * 60)
print("  音频诊断总结")
print("=" * 60)

if errors:
    print(f"\n  ❌ 发现 {len(errors)} 个错误:")
    for e in errors:
        print(f"     - {e}")

if warnings:
    print(f"\n  ⚠ {len(warnings)} 个警告:")
    for w in warnings:
        print(f"     - {w}")

if not errors and not warnings:
    print("\n  ✅ 所有音频测试通过！")

print(f"\n  当前音频设备配置:")
print(f"    AUDIO_DEVICE = {AUDIO_DEVICE}")
try:
    from src.config import TTS_VOICE
    print(f"    TTS_VOICE    = {TTS_VOICE}")
except:
    pass
print()
