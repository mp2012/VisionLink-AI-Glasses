# VisionLink-AI-Glasses
Offline multimodal assistive glasses based on Gemma4 for visually impaired people

## 项目简介
VisionLink 是一款**全离线端侧AI助盲眼镜Demo**，依托Gemma4多模态大模型，实现避障预警、文字朗读、环境描述三大功能。
无需云端联网，低成本硬件即可部署，面向视障人群做普惠无障碍产品，参赛于AI for Good & Gemma On-Device赛道。

## 核心功能
1. 🟢 避障模式：识别障碍物、标注方位与预估距离，语音提醒避险
2. 🟡 文字阅读：提取药盒、路牌、书本文字并实时播报
3. 🔵 场景描述：口语化简述周边环境，辅助日常出行社交

## 环境依赖
```bash
pip install opencv-python ollama pillow


提前拉取模型：ollama pull gemma4:e2b
运行方式

    接入 USB 外接摄像头
    执行源码启动程序
    按键：1/2/3 切换模式，空格拍照识别，ESC 退出

产品规划
PC 为演示原型，后续适配树莓派，整机量产成本控制在 500 元以内，项目采用 MIT 开源协议。
License
MIT
