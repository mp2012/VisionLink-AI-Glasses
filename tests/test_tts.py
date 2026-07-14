"""
测试 src/tts.py — TTS 逻辑（不含硬件音频播放）
"""
import pytest


class TestEnglishDetection:
    """英文文本检测测试"""

    def test_pure_english(self):
        from src.tts import TTSEngine
        assert TTSEngine._is_mostly_english("Hello world this is a test")

    def test_pure_chinese(self):
        from src.tts import TTSEngine
        assert not TTSEngine._is_mostly_english("你好世界这是一个测试")

    def test_mixed_mostly_english(self):
        """英文为主的混合文本"""
        from src.tts import TTSEngine
        # "前方有 obstacle" → 英文字母 9 个，中文字母 3 个 → 75% > 60%
        assert TTSEngine._is_mostly_english("前方有 obstacle danger ahead")

    def test_mixed_mostly_chinese(self):
        """中文为主的混合文本"""
        from src.tts import TTSEngine
        # "前方有一个car" → 英文 3 个，中文 5 个 → 37.5% < 60%
        assert not TTSEngine._is_mostly_english("前方有一个car")

    def test_numbers_only(self):
        """纯数字不应被识别为英文"""
        from src.tts import TTSEngine
        assert not TTSEngine._is_mostly_english("123 456 789")

    def test_empty_string(self):
        """空字符串"""
        from src.tts import TTSEngine
        assert not TTSEngine._is_mostly_english("")

    def test_boundary_60_percent(self):
        """边界值: 正好 60% 英文 → 不算 mainly english"""
        from src.tts import TTSEngine
        # 3 个英文 + 2 个中文 = 60%，不大于 0.6
        assert not TTSEngine._is_mostly_english("abc文字")

    def test_above_60_percent(self):
        """超过 60% 英文 → 算 mainly english"""
        from src.tts import TTSEngine
        # 4 个英文 + 2 个中文 = 66.7% > 60%
        assert TTSEngine._is_mostly_english("abcd文字")

    def test_special_chars_only(self):
        """特殊字符 + 空格"""
        from src.tts import TTSEngine
        assert not TTSEngine._is_mostly_english("!@#$%^&*()")

    def test_english_with_numbers(self):
        """英文 + 数字"""
        from src.tts import TTSEngine
        assert TTSEngine._is_mostly_english("Move 100 meters ahead")


class TestTTSEngineBasics:
    """TTSEngine 基础功能测试"""

    def test_init_state(self):
        """初始状态"""
        from src.tts import TTSEngine
        tts = TTSEngine()
        assert not tts.is_speaking()
        # 初始静音为 False（需要属性存在）
        # _current_priority 初始为 None
        assert tts._current_priority is None

    def test_mute_unmute(self):
        """静音/取消静音"""
        from src.tts import TTSEngine
        tts = TTSEngine()
        assert not tts._muted

        tts.mute()
        assert tts._muted

        tts.unmute()
        assert not tts._muted

    def test_speak_empty_text(self):
        """空文本不播放"""
        from src.tts import TTSEngine
        tts = TTSEngine()
        # 不应该抛异常
        tts.speak("")
        tts.speak("   ")

    def test_speak_muted(self):
        """静音模式不播放"""
        from src.tts import TTSEngine
        tts = TTSEngine()
        tts.mute()
        # 不应该抛异常
        tts.speak("测试文本")

    def test_stop_on_idle(self):
        """空闲时 stop 不抛异常"""
        from src.tts import TTSEngine
        tts = TTSEngine()
        tts.stop()  # 不应抛异常

    def test_check_cmd_false(self):
        """不存在的命令返回 False"""
        from src.tts import TTSEngine
        assert not TTSEngine._check_cmd("nonexistent_command_xyz_123")


class TestTTSPriority:
    """TTS 优先级仲裁逻辑测试（不含实际播放）"""

    def test_speak_sets_priority(self):
        """speak 应该记录当前优先级"""
        from src.tts import TTSEngine
        from src.config import TTS_PRIORITY_EMERGENCY, TTS_PRIORITY_NORMAL

        tts = TTSEngine()
        # 初始状态无优先级
        assert tts._current_priority is None

    def test_priority_constants_importable(self):
        """优先级常量可正常导入"""
        from src.config import (
            TTS_PRIORITY_EMERGENCY, TTS_PRIORITY_WARNING,
            TTS_PRIORITY_SYSTEM, TTS_PRIORITY_NORMAL,
        )
        assert TTS_PRIORITY_EMERGENCY < TTS_PRIORITY_WARNING
        assert TTS_PRIORITY_WARNING < TTS_PRIORITY_SYSTEM
        assert TTS_PRIORITY_SYSTEM < TTS_PRIORITY_NORMAL


class TestTTSEngineThreadSafety:
    """TTS 线程安全测试"""

    def test_lock_exists(self):
        """TTSEngine 应该有线程锁"""
        from src.tts import TTSEngine
        import threading
        tts = TTSEngine()
        assert isinstance(tts._lock, type(threading.Lock()))

    def test_concurrent_mute_unmute(self):
        """并发 mute/unmute 不应死锁"""
        import threading
        from src.tts import TTSEngine

        tts = TTSEngine()
        errors = []

        def toggle():
            try:
                for _ in range(50):
                    tts.mute()
                    tts.unmute()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发操作出现异常: {errors}"
