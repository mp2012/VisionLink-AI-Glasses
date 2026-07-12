# VisionLink-AI-Glasses

An offline multimodal generative AI assistive system based on Gemma 4 for visually impaired individuals.

> 🎬 **Latest Demo Video**: [Click to watch the Bilibili Demo Video](https://www.bilibili.com/video/BV1FmJJ6rEsn/)

---

## Table of Contents

- [VisionLink-AI-Glasses](#visionlink-ai-glasses)
  - [Table of Contents](#table-of-contents)
  - [Project Introduction](#project-introduction)
  - [Core Dimensions \& Highlights](#core-dimensions--highlights)
  - [Functional Modes](#functional-modes)
  - [Dual-Perspective Hardware Architecture \& BOM](#dual-perspective-hardware-architecture--bom)
    - [1. Software Architecture: "Local for Critical Safety, Cloud for Strategic Intel"](#1-software-architecture-local-for-critical-safety-cloud-for-strategic-intel)
    - [2. Hardware Bill of Materials (BOM)](#2-hardware-bill-of-materials-bom)
  - [Project Structure](#project-structure)
    - [Platform Differences](#platform-differences)
  - [Dependencies \& Deployment](#dependencies--deployment)
    - [1. Windows Desktop Version](#1-windows-desktop-version)
    - [2. Jetson Orin Nano Edge Version](#2-jetson-orin-nano-edge-version)
  - [Accessibility Interaction Guide](#accessibility-interaction-guide)
  - [Product Roadmap](#product-roadmap)
  - [License \& Acknowledgments](#license--acknowledgments)

---

## Project Introduction

**VisionLink-AI-Glasses** is a **fully offline, edge-based embodied AI visual compensation system specifically designed for the visually impaired and specialized industries. Powered by edge-optimized Vision-Language Models (VLMs) and a distributed "Edge-Cloud Collaboration" architecture, the project utilizes a highly integrated, split-type wearable design. It delivers high-real-time, absolute privacy, and zero-operational-cost visual assistance in entirely disconnected environments.

By leveraging lightweight multimodal models for efficient on-device inference, the project aims to create inclusive, barrier-free mobility solutions for the visually impaired population. This deeply aligns with the judging criteria of **Google Hackathon Track B: Multimodal**.

---

## Core Dimensions & Highlights

* **🧠 Cognitive-Level Environmental Reasoning**: Breaking away from the limitations of traditional sensors or small models like YOLO. By deploying the Gemma 4 multimodal large model locally, the system doesn't just "see" obstacles; it understands complex spatial causal relationships (e.g., instead of a rigid prompt like *"Bicycle,"* it whispers, *"A shared bicycle has fallen over on the tactile paving ahead, please bypass it to the left"*).
* **🔒 Absolute Privacy & Zero Operational Cost**: Responding to the strict privacy demands of visually impaired users, 100% of the core operational pipeline runs disconnected on the edge. This achieves physical-level privacy lock and guarantees zero-compute operational costs for the enterprise.
* **⚡ Extreme Edge-Side Quantization Acceleration**: Deeply quantized and fine-tuned for the **NVIDIA Jetson Orin Nano (8GB)** memory profile, compressing VLM slow-thinking cycles into seconds. It is paired with a front-loaded fast-track sensor fusion algorithm to balance global spatial planning with millisecond-level emergency obstacle avoidance.
* **🎒 Cyberpunk-Style Split-Wearable Engineering**: Saying no to top-heavy, anti-ergonomic designs. The headwear (glasses side) only retains a micro-camera and ear-clip bone-conduction headphones (total weight < 30g). The motherboard, cooling fan, and PD battery packs are lowered down to the body's load-bearing center, creating a highly recognizable cyberpunk-style Minimum Viable Product (MVP).

---

## Functional Modes

By deeply integrating three distinct modalities—image vision, text comprehension, and audio broadcasting—the project offers five types of accessibility-friendly interactions:

1. **🟢 YOLO Real-Time Obstacle Avoidance Mode**: YOLOv8 real-time detection of surrounding obstacles (person/car/bicycle, etc.), combined with depth distance estimation for precise relative bearing and distance calculation, with tiered voice alerts (danger/warning) to ensure travel safety.
2. **🟡 Text Reading & OCR Mode**: Accurately recognizes text on pillboxes, street signs, and paper books. Optimized and polished by the VLM, it reads aloud in real time, offering perfect support for OCR and foreign language translation.
3. **🔵 Scene Description Mode**: Provides colloquial, humanized summaries of the surrounding environment (stores, pedestrians, road conditions, etc.) to assist users with daily commuting, social interactions, and spatial awareness.
4. **🟣 Face Detection Mode**: Recognizes facial information in the surroundings to assist in social scenarios.
5. **⚪ Visual Q&A Mode**: Ask the VLM questions freely about the current scene for detailed answers.

---

## Dual-Perspective Hardware Architecture & BOM

The project spans a complete full-stack development path, ranging from **PC prototype verification** to an **integrated edge-side prototype**. To balance high flexibility for daily interactions with high robustness for road hazard avoidance, VisionLink innovatively introduces the **"Dual-Perspective Synergy"** hardware solution:

```text
                ┌──────────────────────────────────┐
                │    VisionLink Dual-Perspective   │
                │          Synergy System          │
                └────────────────┬─────────────────┘
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       ▼                                                   ▼

┌──────────────────────┐                            ┌──────────────────────┐
│ First-Person Perspective│                          │ Third-Person Perspective│
│     (POV) Glasses    │                            │     (FOV) Chest Pack │
├──────────────────────┤                            ├──────────────────────┤
│ Head-tracked Micro   │                            │ Orbbec Astra Plus    │
│ Type-C Camera        │                            │ Matrix Depth Camera  │
├──────────────────────┤                            ├──────────────────────┤
│ High-flexibility     │                            │ Robust, low-latency  │
│ Free Angle of View   │                            │ Ground Field of View │
├──────────────────────┤                            ├──────────────────────┤
│ Scene Understanding, │                            │ Real-time 3D Tactile │
│ OCR, Text Reading    │                            │ Paving Avoidance     │
└──────────────────────┘                            └──────────────────────┘
```

### 1. Software Architecture: "Local for Critical Safety, Cloud for Strategic Intel"

* **Edge Brain (Gemma 4 Local Instance)**: Handles high-frequency, high-real-time obstacle avoidance and daily privacy-sensitive scenarios. It is physically isolated and available 100% offline.
* **Cloud Brain (Cloud VLM API)**: Handles low-frequency, high-consumption deep long-text reading or web information retrieval, serving as a strategic backup for the local brain.

### 2. Hardware Bill of Materials (BOM)

| Hardware Module | Reference Image | Core Specifications & System Function |
| :--- | :---: | :--- |
| **First-Person Vision (POV)**<br>Head-tracked Single-Lens Glasses | <img src="images/hardware/1%20(2).jpg" width="180" alt="Micro Camera on Glasses Frame"/> | **Micro Type-C Camera Module**<br>• Ultra-lightweight, clips seamlessly onto regular glasses frames, allowing the viewpoint to follow head movements naturally.<br>• Responsible for flexible interactive scenarios (text OCR, traffic light recognition, specific object identification, and general knowledge Q&A). |
| **Edge Computing Brain** | <img src="images/hardware/1%20(3).jpg" width="180" alt="Jetson Orin Nano"/> | **NVIDIA Jetson Orin Nano Dev Kit (8GB)**<br>• The portable core of the system, housed safely in the backpack/chest pack.<br>• Delivers up to 40 TOPS of AI compute, perfectly running the quantized edge-side large language models. |
| **Third-Person Vision (FOV)**<br>Chest-Mounted Depth Camera | <img src="images/hardware/1%20(7).jpg" width="180" alt="Depth Camera"/> | **Orbbec Astra Plus / Micro HD Camera Assembly**<br>• Embedded and secured into a 4-point tactical chest rig to maintain a stable horizontal viewpoint.<br>• Outputs real-time 3D Depth Maps, dedicated to path navigation, drop-off detection, and low-lying obstacle avoidance. |
| **Audio Output System** | <img src="images/hardware/1%20(4).jpg" width="180" alt="Ear-clip Micro Headphones"/> | **Ear-clip Open-Ear Micro Headphones**<br>• Open-ear design delivers private AI voice feedback without blocking ambient environmental sounds, keeping visually impaired users safe. |
| **Power Supply System** | <img src="images/hardware/1%20(1).jpg" width="180" alt="High-power Power Bank"/> | **High-Output PD Fast-Charging Power Bank (20000mAh / 165W)**<br>• Ergonomic weight distribution design, ensuring over 6 hours of continuous operation for the edge computer under high-throughput inference loads. |
| **Power Decoy Cable** | <img src="images/hardware/1%20(6).jpg" width="180" alt="DC Decoy Cable"/> | **Type-C to DC High-Current Decoy Cable**<br>• Built-in PD fast-charging protocol decoy chip, perfectly regulating and stabilizing the power bank's output voltage to match the Jetson motherboard standards. |

---

## Project Structure

```text
VisionLink/
├── src/                    # Core Source Code (Cross-platform, 10 modules)
│   ├── platform.py         # Platform detection & environment adaptation
│   ├── config.py           # Unified configuration center
│   ├── camera.py           # Dual camera management (POV glasses + FOV chest)
│   ├── detection.py        # YOLOv8 real-time obstacle detection & depth estimation
│   ├── inference.py        # Ollama multimodal inference (Gemma 4)
│   ├── tts.py              # TTS synthesis (Piper > espeak-ng > edge-tts fallback)
│   ├── ui.py               # UI rendering (YOLO overlay, auto-adapts to headless mode)
│   ├── agent.py            # Core controller (state machine / auto mode / YOLO callback)
│   ├── prompts.py          # Prompt template library (CN/EN bilingual)
│   └── orbbec_depth.py     # Orbbec Astra Plus depth camera ctypes wrapper
├── apps/                   # Application Entries (3)
│   ├── desktop.py          # Windows/Linux Desktop GUI full-featured edition
│   ├── headless.py         # Jetson headless mode (evdev global keyboard listener)
│   └── jetson.py           # Jetson terminal keyboard compatible (backward compat)
├── scripts/                # Diagnostic & testing scripts (5)
│   ├── check_system.py     # One-click system comprehensive diagnostic (8 categories)
│   ├── check_camera.py     # Camera scanning & diagnostic
│   └── check_audio.py      # Audio device detection & TTS test
├── start.sh                # One-click launcher (5 modes)
├── archive/                # Legacy iteration history (11 files)
├── assets/                 # Static resources (Fonts, Audio)
├── docs/                   # Technical documentation
├── Log/                    # Runtime logs
├── requirements.txt        # Universal dependencies
└── requirements-jetson.txt # Jetson-specific dependencies
```

### Platform Differences

| Feature | Windows | Jetson |
| :--- | :--- | :--- |
| Model | `gemma4:e2b` | `gemma4:e2b-it-qat` |
| AI Resolution | 448px | 288px |
| Camera Driver | DSHOW, monocular | V4L2, dual-cam (POV ID=0 + FOV ID=2) |
| TTS Engine | PowerShell SAPI5 | Piper (offline) > espeak-ng > edge-tts |
| Audio Device | Default | AB13X USB Audio (plughw:1,0) |
| UI Environment | Full Panel Display | Auto-adapt Headless / GUI debug window |

---

## Dependencies & Deployment

### 1. Windows Desktop Version

```bash
pip install -r requirements.txt
ollama pull gemma4:e2b
python apps/desktop.py
```

### 2. Jetson Orin Nano Edge Version

```bash
pip install -r requirements-jetson.txt
ollama pull gemma4:e2b-it-qat

# Multiple launch modes
./start.sh              # Default: single-cam POV mode
./start.sh dual         # Dual-cam mode (POV + FOV)
./start.sh full         # Full mode (dual-cam + YOLO avoidance)
./start.sh gui          # Headless + GUI debug window
./start.sh desktop      # Desktop GUI mode
```

> 💡 **Note**: Network connection is required ONLY during the initial model pull. During subsequent operations, all multimodal inference, YOLO obstacle avoidance, and speech synthesis pipelines run **100% locally and offline**.

---

## Accessibility Interaction Guide

| Hotkey | Corresponding Functional Mode |
| :--- | :--- |
| **Key 1** | Switch to 【YOLO Obstacle Avoidance Mode】（dual-cam + depth estimation + tiered voice alerts） |
| **Key 2** | Switch to 【Text Reading Mode】（OCR + VLM polishing） |
| **Key 3** | Switch to 【Scene Description Mode】（colloquial environment summary） |
| **Key 4** | Switch to 【Face Detection Mode】 |
| **Key 5** | Switch to 【Visual Q&A Mode】（free-form questions） |
| **Spacebar** | **Trigger Interaction**: Capture Photo → Local VLM Processing → Earphone Audio Broadcast |
| **M Key** | Toggle 【Auto Mode】：timed auto-capture + YOLO avoidance |
| **S Key** | Stop current speech playback |
| **ESC / Q Key** | Exit and safely terminate the program |

> 💡 In headless mode, the system uses **evdev global keyboard listener** (auto-detects `/dev/input/event*` physical keyboard), no window focus required.

---

## Product Roadmap

* [x] **Phase 1 (PC Demo)**: Completed core pipeline execution on PC; validated three core multimodal application modules.
* [x] **Phase 2 (Edge Porting)**: Successfully ported the code stack to **Jetson Orin Nano (8GB)**; achieved memory optimization via INT4/INT8 quantization.
* [x] **Phase 3 (Engineering Refactor)**: Modularized the code architecture; unified cross-platform interfaces and added headless mode support.
* [x] **Phase 4 (Hardware Integration)**: Fabricated the prototype for the head-tracked micro Type-C glasses camera; initially verified POV image capture stability.
* [x] **Phase 5 (YOLO + Dual-Cam Fusion)**: Completed YOLOv8 real-time obstacle avoidance + depth distance estimation + dual-perspective fusion; global keyboard interaction in headless mode.
* [ ] **Phase 6 (Product Enclosure)**: Complete 3D nylon printing for ergonomic wearable kits; seamlessly modify the tactical chest pack for motherboard concealment and passive cooling.
* [ ] **Phase 7 (Vertical Expansion)**: Cross over to neighboring verticals, transferring technologies into network-isolated industrial inspection robotics and healthcare monitoring for elderlies with dementia.

---

## License & Acknowledgments

* This project is licensed under the **MIT License** - see the LICENSE file for details (commercial usage, modifications, and redistribution permitted).
* Special thanks to the **Google Hackathon** for providing the grand stage to showcase our technology.
