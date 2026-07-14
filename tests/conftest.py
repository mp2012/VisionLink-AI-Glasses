"""
测试共享 fixtures 和工具函数
"""
import sys
import os
import pytest

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_linux_fs(tmp_path):
    """模拟 Linux 文件系统环境"""
    return tmp_path


@pytest.fixture
def sample_frame():
    """生成模拟图像帧（480x640 BGR）"""
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)
