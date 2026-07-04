"""
Prompt 模板库
所有与大模型交互的提示词集中管理，支持中英双语。
"""

# ==================== 功能模式名称 ====================
MODE_NAME_LIST = {
    "zh": ["障碍物检测", "文字识别", "人脸检测", "场景介绍", "语音交互"],
    "en": ["Obstacle Detect", "Text Read", "Face Recognize", "Scene Intro", "Chat & Translate"]
}

STATE_NAME_LIST = {
    "zh": ["空闲", "采集", "聆听", "推理", "播报"],
    "en": ["Idle", "Capture", "Listening", "Inferring", "Speaking"]
}

# ==================== 操作指引文本 ====================
GUIDE_TEXT = {
    "zh": {
        "title": "=== 操作指引 ===",
        "key1": "1-5  : 切换功能",
        "key2": "空格 : 立即执行",
        "key3": "L    : 切换语种",
        "key4": "M    : 开关自动模式",
        "key5": "S    : 停止语音",
        "key6": "ESC  : 退出程序",
        "pipe": "流程: 图像 -> AI -> 语音"
    },
    "en": {
        "title": "=== Operation Guide ===",
        "key1": "1-5  : Switch Mode",
        "key2": "Space: Execute Now",
        "key3": "L    : Toggle Language",
        "key4": "M    : Toggle Auto",
        "key5": "S    : Stop Voice",
        "key6": "ESC  : Exit",
        "pipe": "Pipeline: Image -> AI -> Voice"
    }
}

# ==================== 语音提示文本 ====================
TIP_VOICE = {
    "zh": {
        "start": "系统启动完成",
        "auto_on": "自动助手已开启",
        "auto_off": "自动助手已关闭",
        "lang_switch_en": "语音已切换为英文",
        "lang_switch_zh": "语音已切换为中文",
        "stop_voice": "语音已停止",
        "no_voice": "当前无语音",
        "no_change": "画面无变化，减少提醒",
        "cmd_ok": "收到指令，开始执行",
        "mic_fail": "无法识别语音，请重试",
        "exit": "程序即将退出"
    },
    "en": {
        "start": "System started",
        "auto_on": "Auto agent enabled",
        "auto_off": "Auto agent disabled",
        "lang_switch_en": "Voice switched to English",
        "lang_switch_zh": "Voice switched to Chinese",
        "stop_voice": "Voice stopped",
        "no_voice": "No voice playing",
        "no_change": "Same scene, less reminder",
        "cmd_ok": "Command received, executing",
        "mic_fail": "Failed to recognize voice, please try again",
        "exit": "Program exiting"
    }
}

# ==================== 功能 Prompt 库 ====================
PROMPT_LIB = {
    "zh": [
        "你是智能辅助眼镜的障碍物提醒助手，用简短一句话中文提醒道路风险。",
        "你是一个盲人文字读取器。请直接读出画面中物品上的核心中文印刷文字，严禁输出任何问候语、分析过程或多余废话，直接输出正文！",
        "检测画面中的人脸并给出简短提示。",
        "用简短中文介绍眼前风景场景。",
        "结合图片回答问题，支持中英互译，回答简洁。"
    ],
    "en": [
        "You are an obstacle reminder for smart glasses, warn road hazards briefly in English.",
        "Read all printed text in the picture briefly in English. Do not output any notes, just text.",
        "Detect faces and give a short prompt in English.",
        "Briefly introduce the scene in English.",
        "Answer questions with image, support translation, keep answer short in English."
    ]
}

# ==================== Agent 任务调度 Prompt ====================
AGENT_PROMPT = """
You are a multi-modal task scheduler for smart glasses.
Only output number, no extra words:
1=Obstacle detection
2=Read text
3=Recognize face
4=Introduce scene
5=Chat & translate
Separate multiple numbers with comma.
"""

TASK_PLAN_PROMPT = """
Split user command into tasks, only output numbers:
1=Obstacle 2=Read text 3=Face 4=Scene 5=Chat
Split with comma, no extra words.
User command: {user_cmd}
"""
