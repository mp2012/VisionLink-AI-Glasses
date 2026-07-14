"""
Web 预览模块 —— 局域网 MJPEG 推流 + 综合遥测仪表板

使用方式:
    preview = WebPreview(port=5000)
    preview.set_frame_provider(lambda: get_my_combined_frame())
    preview.start()
    ...
    preview.stop()

启动后同一局域网的手机/电脑浏览器打开:
    http://<本机IP>:<port>  → 综合遥测仪表板（唯一页面）
"""

import threading
import time
import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class WebPreview:
    """Flask MJPEG 推流服务器，运行在后台 daemon 线程中"""

    def __init__(self, port: int = 5000):
        self._port = port
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._app = None  # type: flask.Flask | None

        # 帧提供者：由主程序注入，返回 np.ndarray (BGR)
        self._frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()

        # 编码参数
        self._jpeg_quality = 80
        self._target_fps = 15

    # ---- 对外 API ----

    def set_frame_provider(self, provider: Callable[[], Optional[np.ndarray]]):
        """设置帧提供者回调。provider 返回 BGR numpy 数组或 None"""
        self._frame_provider = provider

    @property
    def port(self) -> int:
        return self._port

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """启动 Flask 服务（后台 daemon 线程）"""
        if self._running:
            logger.warning("Web 预览已在运行中")
            return True

        try:
            from flask import Flask, Response, render_template_string
        except ImportError:
            logger.warning("Flask 未安装，Web 预览不可用 (pip install flask)")
            return False

        self._app = Flask(__name__)
        self._app.logger.setLevel(logging.WARNING)  # 静默 Flask 日志

        # 注册路由
        self._setup_routes()

        # 挂载综合遥测仪表板（/ + /api/dashboard）
        try:
            from src.dashboard_status import system_status
            from src.web_dashboard import register_dashboard
            system_status.start_hardware_monitor()
            register_dashboard(self._app)
        except ImportError as e:
            logger.warning(f"仪表板模块导入失败，跳过遥测仪表板挂载: {e}")
        except Exception as e:
            logger.warning(f"仪表板挂载失败: {e}")

        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True, name="WebPreview")
        self._thread.start()
        logger.info(f"Web 预览已启动: http://0.0.0.0:{self._port}")
        return True

    def stop(self):
        """停止 Flask 服务"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._app = None
        logger.info("Web 预览已停止")

    # ---- 内部 ----

    def _setup_routes(self):
        import os as _os
        from flask import Response as _Response, send_from_directory as _send_from_directory

        @self._app.route("/video_feed")
        def video_feed():
            return _Response(self._mjpeg_stream(),
                             mimetype="multipart/x-mixed-replace; boundary=frame")

        @self._app.route("/snapshot/<filename>")
        def get_snapshot(filename):
            snapshots_dir = _os.path.join(_os.getcwd(), "snapshots")
            return _send_from_directory(snapshots_dir, filename)

    def _mjpeg_stream(self):
        """MJPEG 生成器：编码帧 → yield JPEG"""
        import cv2 as _cv2
        frame_interval = 1.0 / self._target_fps
        poll_count = 0
        ok_count = 0
        none_count = 0
        err_count = 0
        warmup_logged = False

        while self._running:
            t0 = time.time()
            poll_count += 1

            frame = None
            if self._frame_provider:
                try:
                    frame = self._frame_provider()
                except Exception as e:
                    err_count += 1
                    if err_count <= 3 or err_count % 100 == 0:
                        logger.warning(f"Web 帧提供者异常 (#{err_count}): {e}")

            if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                try:
                    # 如果帧太大，等比缩放到 1280 宽以内，减少局域网带宽
                    h, w = frame.shape[:2]
                    if w > 1280:
                        scale = 1280.0 / w
                        frame = _cv2.resize(frame, (1280, int(h * scale)))

                    ok, jpeg = _cv2.imencode(".jpg", frame,
                                             [_cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                    if ok:
                        with self._lock:
                            self._latest_jpeg = jpeg.tobytes()
                        ok_count += 1
                        if not warmup_logged:
                            warmup_logged = True
                            logger.info(f"Web 预览开始推流: {frame.shape[1]}x{frame.shape[0]}, "
                                        f"JPEG {len(self._latest_jpeg)} bytes")
                except Exception as e:
                    err_count += 1
                    if err_count <= 3 or err_count % 100 == 0:
                        logger.warning(f"Web JPEG 编码异常 (#{err_count}): {e}")
            else:
                none_count += 1

            # 定期报告统计（每 ~15 秒）
            if poll_count % 225 == 0:
                logger.debug(f"Web 预览统计: {poll_count} 轮询, {ok_count} 帧推送, "
                             f"{none_count} 帧为空, {err_count} 错误")

            with self._lock:
                jpeg = self._latest_jpeg

            if jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

            elapsed = time.time() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            else:
                time.sleep(0.001)

    def _run_server(self):
        """Flask 主循环（在 daemon 线程中运行）"""
        try:
            self._app.run(host="0.0.0.0", port=self._port, threaded=True, debug=False,
                          use_reloader=False)
        except Exception as e:
            if self._running:
                logger.error(f"Web 预览服务异常: {e}")
