# VisionLink - 多模态智能辅助眼镜

全离线端侧多模态 AI 辅助眼镜系统，基于 Google Gemma4 多模态大模型，支持 **Windows 桌面** 和 **Jetson Orin Nano** 双平台。

## 目录结构

```
VisionLink/
├── src/                    # 核心源码（跨平台）
│   ├── platform.py         # 平台检测与环境适配
│   ├── config.py           # 统一配置中心
│   ├── camera.py           # 摄像头管理（DSHOW/V4L2）
│   ├── inference.py        # Ollama 多模态推理
│   ├── tts.py              # TTS 语音合成（SAPI5/espeak）
│   ├── ui.py               # UI 绘制（自动适配无头模式）
│   ├── agent.py            # 自动模式任务调度
│   └── prompts.py          # Prompt 模板库
├── apps/                   # 应用入口
│   ├── desktop.py          # Windows 桌面全功能版
│   └── jetson.py           # Jetson Orin Nano 边缘部署版
├── scripts/                # 测试与工具脚本
├── archive/                # 历史迭代版本
├── assets/                 # 静态资源（字体/音频/图片）
├── docs/                   # 文档
├── requirements.txt        # 通用依赖
└── requirements-jetson.txt # Jetson 专用依赖
```

## 快速启动

### Windows 桌面版
```bash
pip install -r requirements.txt
python apps/desktop.py
```

### Jetson Orin Nano
```bash
pip install -r requirements-jetson.txt
python apps/jetson.py
```

## 功能模式

| 按键 | 功能 |
|------|------|
| 1 | 障碍物检测 |
| 2 | 文字识别 (OCR) |
| 3 | 人脸检测 |
| 4 | 场景描述 |
| 5 | 语音交互 |
| 空格 | 手动触发识别 |
| M | 开关自动模式 |
| L | 中英切换 |
| S | 停止语音 |
| ESC | 退出 |

## 平台差异

| 特性 | Windows | Jetson |
|------|---------|--------|
| 模型 | gemma4:e2b | gemma4:e2b-q4_K_S |
| AI 分辨率 | 448px | 288px |
| 摄像头 | DSHOW, ID=1 | V4L2, 自动遍历 |
| TTS | PowerShell SAPI5 | espeak |
| 蜂鸣 | winsound.Beep | speaker-test |
| UI | 完整面板 | 自动适配无头 |

## License

MIT
