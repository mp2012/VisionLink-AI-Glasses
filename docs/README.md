# VisionLink 技术文档

## 项目概述

**VisionLink-AI-Glasses** 是一款专为视障人士设计的**全离线端侧具身智能视觉代偿系统**，基于 Google Gemma 4 多模态大模型，依托 NVIDIA Jetson Orin Nano (8GB) 边缘设备运行。项目采用 "边云结合" 分布式架构和分体式可穿戴设计，在完全断网环境下实现高实时、绝对隐私、零运营成本的视觉辅助。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    apps/ (应用入口层)                        │
│     desktop.py (桌面 GUI)  │  headless.py (无头模式)         │
│     jetson.py (终端兼容版)                                    │
├─────────────────────────────────────────────────────────────┤
│                     src/ (核心逻辑层)                         │
│  agent.py ── 核心控制中枢（状态机 / 自动模式 / YOLO 回调）     │
│    ├── camera.py      # 双摄像头管理（POV + FOV）            │
│    ├── detection.py   # YOLOv8 实时障碍物检测                │
│    ├── inference.py   # Ollama Gemma4 多模态推理             │
│    ├── tts.py         # TTS 语音合成（三级回退）              │
│    ├── ui.py          # UI 渲染（OpenCV + PIL）              │
│    ├── prompts.py     # Prompt 模板库（中英双语）             │
│    ├── orbbec_depth.py # Orbbec 深度相机 ctypes 封装         │
│    ├── config.py      # 统一配置中心                         │
│    └── platform.py    # 平台检测与环境适配                    │
├─────────────────────────────────────────────────────────────┤
│                   scripts/ (诊断工具层)                       │
│  check_system.py  │  check_camera.py  │  check_audio.py     │
└─────────────────────────────────────────────────────────────┘
```

## 核心模块详解

### 1. `platform.py` — 平台检测

- 通过读取 `/proc/device-tree/model` 检测 NVIDIA Jetson 硬件
- 导出 `IS_JETSON`、`IS_WINDOWS`、`IS_LINUX`、`HAS_DISPLAY` 全局布尔值
- 提供 `get_platform_name()` 函数

### 2. `config.py` — 统一配置中心

| 配置类别 | 关键参数 |
|----------|----------|
| 模型 | `gemma4:e2b-it-qat` (Jetson) / `gemma4:e2b` (Windows) |
| POV 摄像头 | Jetson: ID=0, V4L2, 640x480; Windows: ID=0, DSHOW, 800x600 |
| FOV 摄像头 | Jetson: ID=2, V4L2, 640x480 |
| AI 推理 | Jetson: 288px, 超时 30s; Windows: 448px, 超时 12s |
| YOLO | 置信度 0.5, 6 类目标, 预警 1.5m/危险 0.5m, 播报冷却 2s |
| 音频 | Piper > espeak-ng > edge-tts 三级回退 |
| UI | 面板 320px, 半透明 0.65 |

### 3. `camera.py` — 双摄像头管理

- **`CameraManager`**：通用单摄像头管理器
  - Jetson: GStreamer MJPG pipeline 优先，回退 V4L2 ID 扫描
  - Windows: DSHOW 后端，自动降级
  - 线程安全锁 + 自动重连（最多 5 次）
  - 提供 `crop_roi()` 静态方法裁剪中央区域
- **`DualCameraManager`**：双摄像头协同（POV 镜腿 + FOV 胸前）
  - `open_both()` → `(pov_ok, fov_ok)`
  - 分别提供 `read_pov()` / `read_fov()`

### 4. `detection.py` — YOLO 实时避障

- **`YOLODetector`**：独立线程运行，约 10 FPS
- 检测 6 类目标：人/自行车/汽车/摩托车/公交车/卡车
- 方位描述：左侧/正前方/右侧 + 远处/前方/近处
- 危险分级：danger（人/汽车/摩托车/公交车/卡车）、warning（自行车）
- 播报冷却 2s 防重复
- 回调 `on_detect` 输出结果
- `annotate_frame()` 可视化标注

### 5. `inference.py` — 多模态推理引擎

- **`InferenceEngine`**：封装 Ollama API
  - `image_to_base64()`：JPEG 编码 → Base64
  - `infer()`：同步推理，状态锁防并发
  - Gemma4 thinking 模型兼容：自动提取 `done thinking` 后的最终答案
  - `infer_async()`：异步推理 + Jetson 超时兜底

### 6. `tts.py` — 语音合成

| 平台 | TTS 引擎 | 优先级 |
|------|----------|--------|
| Windows | PowerShell SAPI5 | 默认 |
| Jetson | Piper (离线, 高品质) | 1st |
| Jetson | espeak-ng (离线, 回退) | 2nd |
| Jetson | edge-tts (在线) | 3rd |

- 播报队列：新语音自动中断旧语音
- 音效：快门声 (pic.mp3) / 按键 beep (800Hz) / 告警音 (1200Hz)
- 静音模式支持

### 7. `ui.py` — UI 渲染

- 半透明侧面板 (320px × 65% 透明度)
- 显示：标题、状态指示灯、模式、自动模式状态、语种
- YOLO 检测框叠加（红=危险/黄=警告/绿=普通）
- 中文字体自动检测
- 无头模式自动适配（Null Object Pattern）

### 8. `agent.py` — 核心控制中枢

- **状态机**：IDLE → CAPTURE → LISTEN → INFER → TTS
- **5 种模式**：障碍物检测(1) / 文字识别(2) / 人脸检测(3) / 场景描述(4) / 图文问答(5)
- **手动触发**：空格键 → 拍照 → 推理 → TTS (防抖 0.8s)
- **自动模式**：定时扫描 (Jetson 6s / Windows 5s) → 任务分配 → 去重静默
- **YOLO 回调**：分级语音预警（冷却 2s）
- **语音交互**：speech_recognition + Google STT，最多 3 轮上下文
- 快照保存到 `snapshots/`，日志保存到 `infer_logs/`

### 9. `prompts.py` — Prompt 模板库

- 中英双语，5 种模式的专用提示词
- `AGENT_PROMPT`：自动模式场景分类
- `TASK_PLAN_PROMPT`：语音命令解析
- 操作指引和语音提示文本

### 10. `orbbec_depth.py` — Orbbec 深度相机

- ctypes 封装 `libOrbbecSDK.so` C API
- 封装 Context/Pipeline/Config/Frame 等 SDK 对象
- 深度图 (mm 单位) + 彩色图读取
- 便捷方法：`get_center_depth()` / `get_depth_at_point()` / `get_bbox_depth()`
- 注意：Astra Pro Plus 深度传感器在 SDK 中注册为 IR sensor (type=3)

## 应用入口

### `apps/headless.py` — 主入口（Jetson 无头模式）

- 命令行参数：`--dual` / `--yolo` / `--gui` / `--model`
- **evdev 全局键盘监听**：自动识别 `/dev/input/event*` 物理键盘，无需窗口焦点
- 回退到 termios 非阻塞终端输入
- 支持双摄 + YOLO + GUI 调试窗口

### `apps/desktop.py` — Windows/Linux 桌面版

- 全功能 GUI，单摄像头 POV
- OpenCV `cv2.waitKey()` 按键捕获

### `apps/jetson.py` — 终端兼容版

- 简化版无头模式，单摄像头 POV
- `select.select` 非阻塞终端输入

## 诊断工具

| 脚本 | 功能 |
|------|------|
| `scripts/check_system.py` | 8 大类综合诊断（平台/依赖/Ollama/YOLO/摄像头/音频/模块/字体） |
| `scripts/check_camera.py` | V4L2 设备扫描 + OpenCV 多后端测试 |
| `scripts/check_audio.py` | ALSA/PulseAudio + beep + TTS 全链路测试 |

## 启动方式

```bash
# Jetson 一键启动
./start.sh              # 默认：单摄 POV
./start.sh dual         # 双摄（POV + FOV）
./start.sh full         # 全功能（双摄 + YOLO）
./start.sh gui          # 无头 + GUI 调试
./start.sh desktop      # 桌面 GUI

# Windows
python apps/desktop.py

# 手动启动（高级）
python apps/headless.py --dual --yolo --gui
```

## 数据流

```
POV 摄像头（镜腿）→ 图像帧 ─→ Agent 触发 ─→ Ollama Gemma4 ─→ TTS 播报
                                          └→ 快照保存

FOV 摄像头（胸前）→ 图像帧 ─→ YOLO 检测 ─→ DetectionResult ─→ TTS 避障播报
                                          └→ UI 标注叠加
```

## 平台适配策略

| 特性 | Windows | Jetson Orin Nano |
|------|---------|------------------|
| 模型 | `gemma4:e2b` | `gemma4:e2b-it-qat` |
| AI 分辨率 | 448px | 288px |
| 摄像头驱动 | DSHOW, 单目 | V4L2, 双摄 |
| TTS 引擎 | SAPI5 | Piper/espeak-ng/edge-tts |
| 键盘监听 | cv2.waitKey | evdev 全局监听 |
| UI | 完整面板 | 无头/调试窗口 |

## 深度相机（Orbbec）

Orbbec Astra Pro Plus 通过 C++ SDK v1.10.27 和 ctypes 封装接入：

- SDK 安装路径：`~/.local/lib/libOrbbecSDK.so.1.10.27`
- Python 模块：`src/orbbec_depth.py`
- 分辨率：640x480，单位 mm
- 需要 udev 规则获取 USB 权限

## License

MIT License
