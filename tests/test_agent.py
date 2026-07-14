"""
测试 src/agent.py — 控制中枢逻辑（mock 推理/TTS 引擎）
"""
import time
import pytest
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def mock_infer():
    """Mock 推理引擎"""
    infer = Mock()
    infer.is_busy = False
    infer.image_to_base64.return_value = "fake_base64_string"
    infer.infer.return_value = "推理结果"
    return infer


@pytest.fixture
def mock_tts():
    """Mock TTS 引擎"""
    tts = Mock()
    tts._current_priority = None
    tts._muted = False
    tts.is_speaking.return_value = False
    return tts


@pytest.fixture
def agent(mock_infer, mock_tts):
    """创建可测试的 Agent 实例"""
    from src.agent import Agent
    return Agent(mock_infer, mock_tts, camera_manager=None)


class TestAgentInit:
    """Agent 初始化测试"""

    def test_initial_state(self, agent):
        """初始状态"""
        from src.config import STATE_IDLE
        assert agent.state == STATE_IDLE
        assert not agent.is_busy
        assert not agent.auto_enabled
        assert not agent.yolo_enabled

    def test_initial_mode(self, agent):
        """初始模式为 1"""
        assert agent.current_mode == 1
        assert agent.mode_name == "障碍物检测"

    def test_voice_lang_default_zh(self, agent):
        """默认中文"""
        assert agent._voice_lang == "zh"


class TestModeManagement:
    """模式管理测试"""

    def test_set_valid_mode(self, agent, mock_tts):
        """设置有效模式"""
        agent.set_mode(2)
        assert agent.current_mode == 2
        mock_tts.speak.assert_called_once()
        # 应该使用 SYSTEM 优先级
        call_args = mock_tts.speak.call_args
        assert call_args[1]["priority"] == 2  # TTS_PRIORITY_SYSTEM

    def test_set_invalid_mode(self, agent, mock_tts):
        """无效模式不改变状态"""
        agent.set_mode(0)
        assert agent.current_mode == 1  # 不变
        agent.set_mode(6)
        assert agent.current_mode == 1  # 不变
        mock_tts.speak.assert_not_called()

    def test_mode_switch_yolo_pause(self, agent):
        """模式 1→2 暂停 YOLO，模式 2→1 恢复 YOLO"""
        from src.config import YOLO_PAUSE_EVENT
        YOLO_PAUSE_EVENT.set()  # 初始运行中

        # 1→2 暂停
        agent.set_mode(2)
        assert not YOLO_PAUSE_EVENT.is_set()

        # 2→1 恢复
        agent.set_mode(1)
        assert YOLO_PAUSE_EVENT.is_set()

    def test_mode_name_property(self, agent):
        """模式名称属性"""
        agent._current_mode = 1
        assert agent.mode_name == "障碍物检测"
        agent._current_mode = 5
        # mode_name 来自 MODE_NAMES[4]，根据实际配置可能是"图文问答"或"语音交互"
        assert agent.mode_name in ("语音交互", "图文问答")
        agent._current_mode = 99
        assert agent.mode_name == "未知"


class TestAutoMode:
    """自动模式测试"""

    def test_toggle_auto_on(self, agent, mock_tts):
        """开启自动模式"""
        assert not agent.auto_enabled
        result = agent.toggle_auto()
        assert result is True
        assert agent.auto_enabled
        mock_tts.speak.assert_called_once()

    def test_toggle_auto_off(self, agent):
        """关闭自动模式"""
        agent._auto_enabled = True
        result = agent.toggle_auto()
        assert result is False
        assert not agent.auto_enabled

    def test_should_scan_when_disabled(self, agent):
        """自动模式关闭时不应扫描"""
        assert not agent.should_scan(time.time())

    def test_should_scan_when_enabled_idle(self, agent):
        """自动模式开启且空闲时应扫描"""
        agent._auto_enabled = True
        # 过期的 last_scan 时间
        agent._last_scan_time = 0
        assert agent.should_scan(time.time() + 10)

    def test_should_scan_too_soon(self, agent):
        """自动模式开启但间隔不足不应扫描"""
        from src.config import AGENT_SCAN_INTERVAL
        agent._auto_enabled = True
        now = time.time()
        agent._last_scan_time = now  # 刚刚扫描过
        assert not agent.should_scan(now + AGENT_SCAN_INTERVAL / 2)

    def test_should_scan_when_busy(self, agent):
        """系统忙时不应扫描"""
        agent._auto_enabled = True
        agent._last_scan_time = 0
        agent.infer.is_busy = True
        assert not agent.should_scan(time.time() + 10)


class TestLangSwitch:
    """语言切换测试"""

    def test_set_lang_zh(self, agent, mock_tts):
        agent.set_lang("zh")
        assert agent._voice_lang == "zh"
        mock_tts.speak.assert_called_once()

    def test_set_lang_en(self, agent, mock_tts):
        agent.set_lang("en")
        assert agent._voice_lang == "en"
        mock_tts.speak.assert_called_once()


class TestTriggerHandling:
    """手动触发测试"""

    def test_trigger_initial(self, agent, mock_tts):
        """初始触发"""
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        agent.handle_trigger(frame)
        # 应该播放快门音效
        mock_tts.play_shutter.assert_called_once()

    def test_trigger_debounce(self, agent, mock_tts):
        """防抖：短时间内重复触发被忽略"""
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        agent.handle_trigger(frame)
        mock_tts.play_shutter.assert_called_once()

        # 立即再次触发，防抖应阻止
        mock_tts.reset_mock()
        agent.handle_trigger(frame)
        mock_tts.play_shutter.assert_not_called()

    def test_trigger_when_busy(self, agent, mock_tts):
        """系统忙时手动触发被拒绝"""
        import numpy as np
        agent.infer.is_busy = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 等防抖过去
        agent._last_trigger_time = 0
        mock_tts.reset_mock()
        agent.handle_trigger(frame)
        mock_tts.play_shutter.assert_not_called()


class TestTaskParsing:
    """任务编号解析测试"""

    def test_single_number(self, agent):
        result = agent._parse_task_numbers("3")
        assert result == [3]

    def test_multiple_numbers(self, agent):
        result = agent._parse_task_numbers("1,3,5")
        assert result == [1, 3, 5]

    def test_with_spaces(self, agent):
        result = agent._parse_task_numbers(" 1 , 2 , 4 ")
        assert result == [1, 2, 4]

    def test_out_of_range_filtered(self, agent):
        result = agent._parse_task_numbers("1,6,2,0")
        assert result == [1, 2]  # 6 和 0 被过滤

    def test_invalid_input(self, agent):
        result = agent._parse_task_numbers("hello")
        assert result == []

    def test_empty_string(self, agent):
        result = agent._parse_task_numbers("")
        assert result == []

    def test_duplicates(self, agent):
        result = agent._parse_task_numbers("1,1,2")
        assert result == [1, 1, 2]  # 不去重，保持原样


class TestYOLOCallback:
    """YOLO 避障回调测试"""

    def test_yolo_disabled_no_speak(self, agent, mock_tts):
        """YOLO 关闭时不播报"""
        from src.detection import DetectionResult
        result = DetectionResult(dangers=["前方有人"])
        agent.on_yolo_detect(result)
        mock_tts.speak.assert_not_called()

    def test_yolo_enabled_danger_speak(self, agent, mock_tts):
        """YOLO 开启时危险播报"""
        from src.detection import DetectionResult
        agent._yolo_enabled = True
        result = DetectionResult(dangers=["前方有人"])
        agent.on_yolo_detect(result)
        mock_tts.speak.assert_called_once()
        call_args = mock_tts.speak.call_args
        assert call_args[1]["priority"] == 0  # TTS_PRIORITY_EMERGENCY

    def test_yolo_enabled_warning_priority(self, agent, mock_tts):
        """YOLO 预警使用 WARNING 优先级"""
        from src.detection import DetectionResult
        agent._yolo_enabled = True
        result = DetectionResult(warnings=["左侧有自行车"])
        agent.on_yolo_detect(result)
        mock_tts.speak.assert_called_once()
        call_args = mock_tts.speak.call_args
        assert call_args[1]["priority"] == 1  # TTS_PRIORITY_WARNING

    def test_yolo_cooldown_danger(self, agent, mock_tts):
        """危险冷却时间内不重复播报（3秒）"""
        from src.detection import DetectionResult
        agent._yolo_enabled = True
        result = DetectionResult(dangers=["前方有人"])

        agent.on_yolo_detect(result)
        assert mock_tts.speak.call_count == 1

        # 立即再次触发，应在冷却期内
        agent.on_yolo_detect(result)
        assert mock_tts.speak.call_count == 1  # 无新增

    def test_yolo_cooldown_warning(self, agent, mock_tts):
        """预警冷却时间内不重复播报（8秒）"""
        from src.detection import DetectionResult
        agent._yolo_enabled = True
        agent._yolo_last_announce = time.time() - 4  # 4秒前播报过
        result = DetectionResult(warnings=["左侧有自行车"])

        agent.on_yolo_detect(result)
        mock_tts.speak.assert_not_called()  # 还在 8 秒冷却内

    def test_yolo_empty_message_no_speak(self, agent, mock_tts):
        """空播报消息不应触发"""
        from src.detection import DetectionResult
        agent._yolo_enabled = True
        result = DetectionResult()  # 无警告无危险
        agent.on_yolo_detect(result)
        mock_tts.speak.assert_not_called()

    def test_toggle_yolo(self, agent):
        """开关 YOLO"""
        assert not agent.yolo_enabled
        result = agent.toggle_yolo()
        assert result is True
        assert agent.yolo_enabled
        result = agent.toggle_yolo()
        assert result is False
        assert not agent.yolo_enabled


class TestObstacleKey:
    """障碍物去重 key 提取测试"""

    def test_front_key(self, agent):
        assert agent._extract_obstacle_key("危险！正前方近处约800毫米有人") == "正前方"

    def test_left_key(self, agent):
        assert agent._extract_obstacle_key("注意，左侧有自行车") == "左侧"

    def test_right_key(self, agent):
        assert agent._extract_obstacle_key("危险！右侧有车") == "右侧"

    def test_fallback_key(self, agent):
        """不含方向时的回退"""
        key = agent._extract_obstacle_key("前面有障碍物")
        assert len(key) > 0  # 返回前 6 个字符


class TestEventCallbacks:
    """事件回调测试"""

    def test_register_and_emit(self, agent):
        callback = Mock()
        agent.on("test_event", callback)
        agent._emit("test_event", "arg1", "arg2")
        callback.assert_called_once_with("arg1", "arg2")

    def test_emit_no_listener(self, agent):
        """无监听时 emit 不抛异常"""
        agent._emit("nonexistent_event")

    def test_callback_exception_handled(self, agent):
        """回调异常被捕获"""
        def bad_callback(*args):
            raise RuntimeError("test error")
        agent.on("bad_event", bad_callback)
        # 不应抛异常
        agent._emit("bad_event")


class TestShutdown:
    """关闭流程测试"""

    def test_shutdown_stops_tts(self, agent, mock_tts):
        agent.shutdown()
        mock_tts.stop.assert_called_once()
        assert not agent.auto_enabled
        assert not agent.yolo_enabled
