"""
测试 src/platform.py — 平台检测逻辑
"""
import os
import platform as _platform
import pytest
from unittest.mock import patch, mock_open


class TestPlatformDetection:
    """平台检测测试"""

    def test_imports_without_error(self):
        """platform 模块可正常导入"""
        from src import platform as p
        assert hasattr(p, "IS_JETSON")
        assert hasattr(p, "IS_WINDOWS")
        assert hasattr(p, "IS_LINUX")
        assert hasattr(p, "HAS_DISPLAY")

    def test_is_jetson_is_boolean(self):
        """IS_JETSON 必须是布尔值"""
        from src.platform import IS_JETSON
        assert isinstance(IS_JETSON, bool)

    def test_is_windows_is_boolean(self):
        """IS_WINDOWS 必须是布尔值"""
        from src.platform import IS_WINDOWS
        assert isinstance(IS_WINDOWS, bool)

    def test_mutual_exclusion(self):
        """一个平台不能同时是 Jetson 和 Windows"""
        from src.platform import IS_JETSON, IS_WINDOWS
        assert not (IS_JETSON and IS_WINDOWS), "不能同时是 Jetson 和 Windows"

    def test_get_platform_name_returns_string(self):
        """get_platform_name 返回非空字符串"""
        from src.platform import get_platform_name
        name = get_platform_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_platform_name_matches_platform(self):
        """get_platform_name 对应正确的平台"""
        from src.platform import get_platform_name, IS_JETSON, IS_WINDOWS, IS_LINUX
        name = get_platform_name()
        if IS_JETSON:
            assert "jetson" in name.lower()
        elif IS_WINDOWS:
            assert "windows" in name.lower()

    @pytest.mark.parametrize("machine,model_exists,model_content,expected", [
        # Jetson: device-tree 含 NVIDIA/Jetson 标识
        ("aarch64", True, "NVIDIA Jetson Orin Nano Developer Kit", True),
        # Jetson: 无 device-tree 但 aarch64 保底
        ("aarch64", False, "", True),
        # 非 Jetson: device-tree 含 Jetson 标识 → 判为 Jetson（合法逻辑：硬件标识优先）
        ("x86_64", True, "NVIDIA Jetson Orin Nano Developer Kit", True),
        # 非 Jetson: 无 device-tree，x86_64 不满足保底条件
        ("x86_64", False, "", False),
        # armv7l 树莓派: 非 Jetson
        ("armv7l", False, "", False),
        # device-tree 存在但非 NVIDIA（普通 ARM 开发板）
        ("aarch64", True, "Raspberry Pi 5 Model B", True),  # aarch64 保底
    ])
    def test_check_jetson_logic(self, monkeypatch, machine, model_exists, model_content, expected):
        """
        测试 _check_jetson() 判定逻辑：
        1. device-tree model 含 NVIDIA/Jetson → 立即 True
        2. device-tree 不存在 → 依靠 platform.machine()=="aarch64" 保底
        """
        monkeypatch.setattr(_platform, "machine", lambda: machine)
        monkeypatch.setattr(_platform, "system", lambda: "Linux")
        monkeypatch.setattr(os.path, "exists", lambda p: (
            model_exists if p == "/proc/device-tree/model" else False
        ))

        if model_exists and model_content:
            monkeypatch.setattr("builtins.open", mock_open(read_data=model_content), raising=False)

        import importlib
        import src.platform as sp
        importlib.reload(sp)

        assert sp.IS_JETSON == expected

    def test_has_display_on_windows(self, monkeypatch):
        """Windows 上 HAS_DISPLAY 始终为 True"""
        monkeypatch.setattr(_platform, "system", lambda: "Windows")
        monkeypatch.setattr(os, "environ", {})

        import importlib
        import src.platform as sp
        importlib.reload(sp)
        assert sp.HAS_DISPLAY is True
