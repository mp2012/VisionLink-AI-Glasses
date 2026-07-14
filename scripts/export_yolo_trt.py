"""
YOLOv8n TensorRT 引擎导出脚本

将 yolov8n.pt 转换为 TensorRT FP16 引擎，在 Jetson Orin Nano 上获得显著加速。
导出过程必须在 Jetson 设备上运行（引擎与 GPU 架构绑定）。

使用方法:
    cd /home/seeed/AI/VisionLink
    python scripts/export_yolo_trt.py                     # 使用默认路径
    python scripts/export_yolo_trt.py --model yolov8n.pt --output yolov8n.engine  # 自定义路径

前置依赖:
    - PyTorch for Jetson (预装在 JetPack)
    - ultralytics >= 8.0.0 (pip install ultralytics)
    - TensorRT (预装在 JetPack)
    - onnx (pip install onnx)

预期效果:
    - PyTorch (.pt): ~50-80ms / 帧
    - TensorRT FP16 (.engine): ~5-15ms / 帧
    - 加速比约 5-10x
"""

import os
import sys
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.platform import IS_JETSON

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} | {levelname:8} | {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
)
logger = logging.getLogger("export_trt")


def check_cuda():
    """检查 CUDA / TensorRT 环境"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        logger.info(f"PyTorch CUDA: {'可用' if cuda_available else '不可用'}")
        logger.info(f"GPU 设备: {device_name}")
        return cuda_available
    except ImportError:
        logger.error("PyTorch 未安装")
        return False


def check_tensorrt():
    """检查 TensorRT 是否可用"""
    try:
        import tensorrt
        logger.info(f"TensorRT 版本: {tensorrt.__version__}")
        return True
    except ImportError:
        logger.error("TensorRT Python 绑定未安装，请检查 JetPack 环境")
        return False


def benchmark_model(model_path: str, n_runs: int = 50) -> tuple:
    """测试模型推理速度"""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(model_path)
    dummy_input = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    # 预热
    for _ in range(10):
        model(dummy_input, verbose=False)

    # 计时
    start = time.perf_counter()
    for _ in range(n_runs):
        model(dummy_input, verbose=False)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_runs) * 1000
    fps = n_runs / elapsed
    return avg_ms, fps


def export_to_tensorrt(model_path: str, output_path: str) -> bool:
    """
    使用 ultralytics 内置的 TensorRT 导出功能

    ultralytics 内部流程:
    1. PyTorch → ONNX
    2. ONNX → TensorRT Engine (FP16)
    """
    from ultralytics import YOLO

    logger.info(f"加载 PyTorch 模型: {model_path}")
    model = YOLO(model_path)

    # 先在 PyTorch 上跑一次确认模型正常
    logger.info("PyTorch 基准测试...")
    pt_avg_ms, pt_fps = benchmark_model(model_path)
    logger.info(f"PyTorch 基准: {pt_avg_ms:.1f}ms / 帧 ({pt_fps:.1f} FPS)")

    # 导出为 TensorRT engine
    logger.info("开始导出 TensorRT 引擎（这个过程可能需要几分钟）...")
    logger.info("步骤: PyTorch → ONNX → TensorRT FP16 Engine")

    try:
        export_start = time.time()
        model.export(
            format="engine",       # 目标格式: TensorRT
            half=True,             # FP16 精度
            imgsz=640,             # 输入尺寸
            device=0,              # 使用 GPU 0
            workspace=4,           # TensorRT workspace (GB)
            verbose=False,
        )
        export_elapsed = time.time() - export_start

        # 导出后的文件名（ultralytics 自动生成）
        # 如果 output_path 指定了路径，需要移动文件
        base_name = os.path.splitext(model_path)[0]
        auto_output = f"{base_name}.engine"

        if auto_output != output_path:
            if os.path.exists(auto_output):
                os.rename(auto_output, output_path)
                logger.info(f"引擎已移动: {auto_output} → {output_path}")

        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"导出成功! 耗时 {export_elapsed:.1f}s, 文件大小 {size_mb:.1f}MB")
        else:
            logger.error(f"导出失败: 未生成引擎文件 {output_path}")
            return False

    except Exception as e:
        logger.error(f"导出异常: {e}")
        logger.info("提示: 若遇到 ONNX 导出问题，请确保已安装 onnx 和 onnxsim:")
        logger.info("  pip install onnx onnxsim")
        return False

    # TensorRT 基准测试
    logger.info("TensorRT 基准测试...")
    try:
        trt_avg_ms, trt_fps = benchmark_model(output_path)
        speedup = pt_avg_ms / trt_avg_ms
        logger.info(f"TensorRT 基准: {trt_avg_ms:.1f}ms / 帧 ({trt_fps:.1f} FPS)")
        logger.info(f"加速比: {speedup:.1f}x")
        logger.info(f"检测延时: {pt_avg_ms:.0f}ms → {trt_avg_ms:.0f}ms (节省 {pt_avg_ms - trt_avg_ms:.0f}ms)")
    except Exception as e:
        logger.warning(f"TensorRT 基准测试失败（不影响导出结果）: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n TensorRT 引擎导出工具")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="PyTorch 模型路径 (默认: yolov8n.pt)"
    )
    parser.add_argument(
        "--output", type=str, default="yolov8n.engine",
        help="TensorRT 引擎输出路径 (默认: yolov8n.engine)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新导出（覆盖已有引擎文件）"
    )
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("YOLOv8n → TensorRT 引擎导出工具")
    logger.info(f"平台: {'Jetson Orin Nano' if IS_JETSON else '非 Jetson'}")
    logger.info(f"输入: {args.model}")
    logger.info(f"输出: {args.output}")
    logger.info("=" * 55)

    # 平台检查
    if not IS_JETSON:
        logger.warning("当前非 Jetson 平台。TensorRT 引擎与 GPU 架构绑定，")
        logger.warning("在其他平台导出的引擎不能在 Jetson 上使用。")
        logger.warning("请在 Jetson Orin Nano 上运行此脚本。")
        choice = input("继续导出? (y/N): ").strip().lower()
        if choice != "y":
            logger.info("已取消")
            return

    # 模型文件检查
    if not os.path.exists(args.model):
        logger.error(f"模型文件不存在: {args.model}")
        logger.info("请先下载 YOLOv8n 模型:")
        logger.info("  wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt")
        return

    # 已存在检查
    if os.path.exists(args.output) and not args.force:
        logger.info(f"引擎文件已存在: {args.output} ({os.path.getsize(args.output) / 1024:.1f}KB)")
        choice = input("重新导出? (y/N): ").strip().lower()
        if choice != "y":
            logger.info("使用 --force 参数可跳过此确认")
            return

    # 环境检查
    if not check_cuda():
        logger.error("CUDA 不可用，无法导出 TensorRT 引擎")
        return

    if not check_tensorrt():
        logger.error("TensorRT 不可用，请检查 JetPack 环境")
        return

    # 执行导出
    success = export_to_tensorrt(args.model, args.output)

    if success:
        logger.info("=" * 55)
        logger.info("导出完成! 引擎文件: " + args.output)
        logger.info("请更新 config.py 中的 YOLO_ENGINE_PATH 配置，")
        logger.info("或确保引擎文件与 yolov8n.pt 在同一目录下（自动加载）。")
        logger.info("")
        logger.info("使用建议:")
        logger.info("  1. 引擎文件与 GPU 架构绑定，更换 Jetson 设备需重新导出")
        logger.info("  2. 更新 JetPack / TensorRT 版本后需重新导出")
        logger.info("  3. 删除 .engine 文件即回退到 PyTorch 模式")
        logger.info("=" * 55)


if __name__ == "__main__":
    main()
