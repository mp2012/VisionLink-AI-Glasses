"""
测试 src/prompts.py — Prompt 模板库完整性
"""
import pytest
from src.prompts import (
    MODE_NAME_LIST, STATE_NAME_LIST, GUIDE_TEXT,
    TIP_VOICE, PROMPT_LIB, AGENT_PROMPT, TASK_PLAN_PROMPT
)


class TestPromptLibrary:
    """Prompt 库完整性测试"""

    # ==================== MODE_NAME_LIST ====================

    def test_mode_names_has_zh_and_en(self):
        """模式名称必须支持中英文"""
        assert "zh" in MODE_NAME_LIST
        assert "en" in MODE_NAME_LIST

    def test_mode_names_count_is_5(self):
        """每个语言应有 5 个模式"""
        for lang in ("zh", "en"):
            assert len(MODE_NAME_LIST[lang]) == 5, f"{lang} 模式数应为 5"

    def test_mode_names_all_non_empty(self):
        """模式名称不能为空"""
        for lang in ("zh", "en"):
            for name in MODE_NAME_LIST[lang]:
                assert isinstance(name, str) and len(name) > 0

    # ==================== STATE_NAME_LIST ====================

    def test_state_names_has_zh_and_en(self):
        """状态名称有中英文"""
        assert set(STATE_NAME_LIST.keys()) == {"zh", "en"}

    def test_state_names_count_is_5(self):
        """应有 5 个状态"""
        for lang in ("zh", "en"):
            assert len(STATE_NAME_LIST[lang]) == 5

    # ==================== GUIDE_TEXT ====================

    def test_guide_text_keys(self):
        """操作指引包含所有必要的键"""
        required = {"title", "key1", "key2", "key3", "key4", "key5", "key6", "pipe"}
        for lang in ("zh", "en"):
            for key in required:
                assert key in GUIDE_TEXT[lang], f"{lang} 缺少 {key}"

    # ==================== TIP_VOICE ====================

    def test_tip_voice_keys(self):
        """语音提示包含所有必要的键"""
        required = {
            "start", "auto_on", "auto_off", "lang_switch_en", "lang_switch_zh",
            "stop_voice", "no_voice", "no_change", "cmd_ok", "mic_fail", "exit"
        }
        for lang in ("zh", "en"):
            for key in required:
                assert key in TIP_VOICE[lang], f"{lang} 缺少语音提示键: {key}"

    def test_tip_voice_non_empty(self):
        """语音提示不能为空"""
        for lang in ("zh", "en"):
            for key, val in TIP_VOICE[lang].items():
                assert isinstance(val, str) and len(val) > 0, f"{lang}.{key} 为空"

    # ==================== PROMPT_LIB ====================

    def test_prompt_lib_count(self):
        """每个语言应有 5 个功能 Prompt"""
        for lang in ("zh", "en"):
            assert len(PROMPT_LIB[lang]) == 5

    def test_prompt_lib_non_empty(self):
        """Prompt 模板不能为空"""
        for lang in ("zh", "en"):
            for i, prompt in enumerate(PROMPT_LIB[lang]):
                assert isinstance(prompt, str) and len(prompt) > 10, \
                    f"{lang}[{i}] 太短: {prompt}"

    def test_prompt_mode1_mentions_obstacle(self):
        """模式 1 的 Prompt 应提到障碍物"""
        assert "障碍" in PROMPT_LIB["zh"][0]
        assert "obstacle" in PROMPT_LIB["en"][0].lower()

    def test_prompt_mode2_mentions_text(self):
        """模式 2 的 Prompt 应提到文字识别"""
        assert "文字" in PROMPT_LIB["zh"][1]
        assert "text" in PROMPT_LIB["en"][1].lower()

    # ==================== AGENT_PROMPT ====================

    def test_agent_prompt_mentions_numbers(self):
        """Agent Prompt 应该引用数字 1-5"""
        agent = AGENT_PROMPT.lower()
        for i in range(1, 6):
            assert str(i) in agent, f"AGENT_PROMPT 缺少数字 {i}"

    # ==================== TASK_PLAN_PROMPT ====================

    def test_task_plan_has_placeholder(self):
        """任务规划 Prompt 必须包含 {user_cmd} 占位符"""
        assert "{user_cmd}" in TASK_PLAN_PROMPT

    def test_task_plan_formats_correctly(self):
        """TASK_PLAN_PROMPT 格式化不抛异常"""
        formatted = TASK_PLAN_PROMPT.format(user_cmd="前方有什么")
        assert "前方有什么" in formatted
