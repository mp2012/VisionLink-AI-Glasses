"""
VisionLink 综合遥测仪表板 —— 路由 + 页面

在现有 Flask app 上挂载 / 路由（唯一页面），
复用已有的 /video_feed 视频流。

接入方式:
    from web_dashboard import register_dashboard
    register_dashboard(app)
"""

import logging
import socket
from flask import Response, jsonify

logger = logging.getLogger(__name__)

# 获取本地首选 IP 供页脚调试显示
def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

_LOCAL_IP = _get_local_ip()

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisionLink 遥测</title>
<style>
  :root {
    --bg: #0a0e14; --card: #12171f; --border: #1e2a38;
    --text: #c0c8d4; --muted: #5c6e82; --dim: #3a4555;
    --green: #2ea043; --red: #da3633; --yellow: #d2991d; --blue: #58a6ff;
    --orange: #f0883e; --cyan: #39d2c0;
    --p0-glow: rgba(218,54,51,.6);
  }
  *,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
  body{
    background:var(--bg); color:var(--text);
    font-family:"SF Mono","Cascadia Code","JetBrains Mono","Consolas",monospace;
    font-size:11px; line-height:1.55;
    overflow-x:hidden;
  }
  /* ===== Header ===== */
  .header{
    display:flex; align-items:center; justify-content:space-between;
    padding:6px 14px; background:var(--card); border-bottom:1px solid var(--border);
    flex-wrap:wrap; gap:4px;
  }
  .header h1{font-size:12px;font-weight:600;color:var(--cyan);letter-spacing:.5px}
  .h-badges{display:flex;gap:6px;flex-wrap:wrap}
  .h-badge{
    font-size:10px;padding:2px 7px;border-radius:3px;font-weight:500;
    white-space:nowrap;letter-spacing:.3px;
  }
  .h-badge.live{background:rgba(46,160,67,.15);color:var(--green);border:1px solid rgba(46,160,67,.25)}
  .h-badge.off{background:rgba(90,100,110,.12);color:var(--muted);border:1px solid var(--dim)}
  /* ===== Layout ===== */
  .main{display:flex;gap:8px;padding:8px;height:calc(100vh - 40px)}
  .col-left{flex:2 1 65%;min-width:360px;display:flex;flex-direction:column;gap:8px}
  .col-right{flex:1 1 35%;min-width:280px;display:flex;flex-direction:column;gap:8px;overflow-y:auto}
  /* ===== Panels ===== */
  .panel{
    border:1px solid var(--border); border-radius:4px;
    background:var(--card); overflow:hidden;
  }
  .panel.vid{flex:1;display:flex;flex-direction:column}
  .panel.vid img{display:block;width:100%;min-height:0;flex:1;object-fit:contain;background:#000}
  .p-title{
    font-size:10px;font-weight:600;padding:4px 10px;
    background:rgba(0,0,0,.25); border-bottom:1px solid var(--border);
    color:var(--muted); text-transform:uppercase; letter-spacing:.8px;
    display:flex; justify-content:space-between; align-items:center;
  }
  .p-title .stale{font-size:9px;color:var(--yellow);display:none}
  .p-body{padding:6px 10px}
  /* ===== Hardware Grid ===== */
  .hw-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 10px}
  .hw-item{display:flex;align-items:center;gap:6px}
  .hw-label{color:var(--muted);font-size:10px;width:28px;flex-shrink:0}
  .hw-bar-wrap{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
  .hw-bar{display:block;height:100%;border-radius:2px;transition:width .6s ease}
  .hw-bar.gpu{background:var(--cyan)}
  .hw-bar.cpu{background:var(--blue)}
  .hw-bar.mem{background:var(--orange)}
  .hw-val{font-size:10px;width:32px;text-align:right;flex-shrink:0;color:var(--text)}
  .hw-temp{font-size:10px;width:38px;text-align:right;flex-shrink:0}
  .hw-temp.cool{color:var(--green)} .hw-temp.warm{color:var(--yellow)} .hw-temp.hot{color:var(--red)}
  /* ===== Hardware section divider ===== */
  .hw-section{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.6px;padding:6px 0 2px 0;margin-top:2px;border-top:1px solid var(--border)}
  .hw-section:first-child{border-top:none;margin-top:0;padding-top:0}
  /* ===== Status Dots ===== */
  .status-row{display:flex;flex-wrap:wrap;gap:12px;padding-top:2px}
  .status-dot{display:flex;align-items:center;gap:5px;font-size:10px}
  .s-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
  .s-dot.on{background:var(--green);box-shadow:0 0 4px rgba(46,160,67,.5)}
  .s-dot.off{background:var(--dim)}
  .s-dot.paused{background:var(--yellow);animation:blink 1.2s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
  @keyframes infer-pulse{0%,100%{opacity:1}50%{opacity:.35}}
  /* ===== Log Lists ===== */
  .log-list{display:flex;flex-direction:column;gap:2px;max-height:220px;overflow-y:auto}
  .log-entry{
    font-size:11px;padding:4px 7px;border-radius:2px;
    display:flex;gap:6px;align-items:flex-start;line-height:1.5;
    border-left:2px solid transparent;
  }
  .log-time{color:var(--dim);white-space:nowrap;flex-shrink:0;font-size:9px}
  .log-prio{font-weight:700;width:20px;flex-shrink:0;text-align:center;font-size:10px}
  .log-text{flex:1;word-break:break-all}
  .log-lat{font-size:9px;color:var(--muted);text-align:right;flex-shrink:0}
  /* inference log variants */
  .log-entry.inf-ok{border-left-color:var(--green)}
  .log-entry.inf-fail{border-left-color:var(--red);color:var(--muted)}
  /* detection log variants */
  .log-entry.det-p0{
    border-left:3px solid var(--red); color:var(--red); font-weight:600;
    background:rgba(218,54,51,.1);
    animation:p0-pulse 1.5s ease-in-out infinite;
  }
  .log-entry.det-p1{border-left-color:var(--yellow)}
  @keyframes p0-pulse{
    0%,100%{background:rgba(218,54,51,.12);border-left-color:var(--red)}
    50%{background:rgba(218,54,51,.04);border-left-color:rgba(218,54,51,.35)}
  }
  /* TTS log variants */
  .log-entry.tts-p0{
    background:rgba(218,54,51,.12); border-left:3px solid var(--red); font-weight:600;
    box-shadow:inset 0 0 16px rgba(218,54,51,.08);
    animation:tts-p0-pulse 2s ease-in-out infinite;
  }
  @keyframes tts-p0-pulse{
    0%,100%{background:rgba(218,54,51,.14)}
    50%{background:rgba(218,54,51,.05)}
  }
  .log-entry.tts-p1{background:rgba(210,153,29,.06);border-left-color:var(--yellow)}
  .log-entry.tts-p2{background:rgba(136,136,136,.04);border-left-color:var(--muted)}
  .log-entry.tts-p3{border-left-color:var(--green)}
  .log-engine{font-size:8px;color:var(--dim);flex-shrink:0;text-align:right}
  /* ===== Health badge ===== */
  .h-badge.health-ok{background:rgba(46,160,67,.15);color:var(--green);border:1px solid rgba(46,160,67,.3)}
  .h-badge.health-warn{background:rgba(210,153,29,.12);color:var(--yellow);border:1px solid rgba(210,153,29,.3)}
  .h-badge.health-crit{background:rgba(218,54,51,.15);color:var(--red);border:1px solid rgba(218,54,51,.3);animation:blink 1.2s infinite}
  /* ===== Empty state ===== */
  .log-empty{font-size:10px;color:var(--dim);padding:8px 6px;text-align:center}
  /* ===== Snapshot panel ===== */
  .panel.snap{flex-shrink:0}
  .panel.snap img{display:block;width:100%;height:auto;max-height:300px;object-fit:contain;background:#000}
  .snap-empty{font-size:10px;color:var(--dim);padding:16px;text-align:center}
  .snap-meta{font-size:9px;color:var(--muted);padding:2px 8px;display:flex;justify-content:space-between}
  .snap-meta .snap-mode{font-weight:600}
  .snap-meta .snap-mode.det{margin-left:auto;color:var(--red)}
  .snap-meta .snap-mode.recog{color:var(--yellow)}
  .snap-meta .snap-mode.face{color:var(--blue)}
  .snap-meta .snap-mode.scene{color:var(--cyan)}
  .snap-meta .snap-mode.qa{color:var(--green)}
  /* ===== Footer ===== */
  .footer{
    text-align:center;padding:3px;font-size:9px;color:var(--dim);
    border-top:1px solid var(--border);background:var(--card);
  }
  /* ===== Responsive ===== */
  @media(max-width:750px){
    .main{flex-direction:column;height:auto}
    .col-left,.col-right{flex:1 1 auto;min-width:100%}
    .col-left{height:40vh}
    .hw-grid{grid-template-columns:1fr}
    .log-list{max-height:140px}
  }
</style>
</head>
<body>
<div class="header">
  <h1>VISIONLINK 遥测</h1>
  <div class="h-badges">
    <span class="h-badge live" id="conn-badge">● ONLINE</span>
    <span class="h-badge" id="health-badge" style="background:var(--card);color:var(--muted)">系统 --</span>
    <span class="h-badge" id="mode-badge" style="background:var(--card);color:var(--muted)">--</span>
    <span class="h-badge" id="yolo-badge" style="background:var(--card);color:var(--muted)">YOLO --</span>
    <span class="h-badge" id="ollama-badge" style="background:var(--card);color:var(--muted)">Ollama --</span>
    <span class="h-badge" id="infer-badge" style="display:none;background:rgba(88,166,255,.12);color:var(--blue);animation:infer-pulse 1.2s infinite">◉ 识别中...</span>
    <span class="h-badge" style="font-size:9px;color:var(--dim)" id="clock">--</span>
  </div>
</div>

<div class="main">
  <div class="col-left">
    <div class="panel vid">
      <div class="p-title">POV + FOV + 深度 <span class="stale" id="vid-stale">⚠ 无信号</span></div>
      <img src="/video_feed" alt="CAM" id="cam-img"
           onerror="document.getElementById('vid-stale').style.display='inline'"
           onload="document.getElementById('vid-stale').style.display='none'">
    </div>

    <div class="panel snap">
      <div class="p-title">最近触发快照 <span id="snap-age" style="font-size:9px;color:var(--dim)">暂无快照</span></div>
      <div class="snap-empty" id="snap-empty">等待首次推理触发</div>
      <img id="snap-img" style="display:none" src="">
      <div class="snap-meta" id="snap-meta" style="display:none"></div>
    </div>
  </div>

  <div class="col-right">
    <div class="panel">
      <div class="p-title">系统状态</div>
      <div class="p-body">
        <div class="hw-section">资源使用</div>
        <div class="hw-grid">
          <div class="hw-item"><span class="hw-label">GPU</span><span class="hw-bar-wrap"><span class="hw-bar gpu" id="bar-gpu" style="width:0%"></span></span><span class="hw-val" id="val-gpu">0%</span></div>
          <div class="hw-item"><span class="hw-label">CPU</span><span class="hw-bar-wrap"><span class="hw-bar cpu" id="bar-cpu" style="width:0%"></span></span><span class="hw-val" id="val-cpu">0%</span></div>
          <div class="hw-item"><span class="hw-label">MEM</span><span class="hw-bar-wrap"><span class="hw-bar mem" id="bar-mem" style="width:0%"></span></span><span class="hw-val" id="val-mem">0%</span></div>
          <div class="hw-item"><span class="hw-label">TEMP</span><span></span><span class="hw-temp cool" id="val-temp">--°C</span></div>
        </div>
        <div class="hw-section">服务状态</div>
        <div class="status-row">
          <div class="status-dot"><span class="s-dot off" id="dot-yolo"></span>YOLO</div>
          <div class="status-dot"><span class="s-dot off" id="dot-depth"></span>深度</div>
          <div class="status-dot"><span class="s-dot off" id="dot-ollama"></span>Ollama</div>
          <div class="status-dot"><span class="s-dot off" id="dot-tts"></span>TTS</div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="p-title">识别结果 <span style="font-size:9px;color:var(--dim)" id="inf-count">0</span></div>
      <div class="p-body" style="padding:2px 0">
        <div class="log-list" id="inf-list"><div class="log-empty">暂无识别记录</div></div>
      </div>
    </div>

    <div class="panel">
      <div class="p-title">语音播报 <span style="font-size:9px;color:var(--dim)" id="tts-count">0</span></div>
      <div class="p-body" style="padding:2px 0">
        <div class="log-list" id="tts-list"><div class="log-empty">暂无播报记录</div></div>
      </div>
    </div>
  </div>
</div>

<div class="footer">VisionLink Telemetry Dashboard · 动态平滑轮询 · <span id="footer-ip">--</span></div>

<script>
(function(){
  // DOM refs
  var $ = function(id){return document.getElementById(id)};
  var CONN = $('conn-badge'), HEALTH = $('health-badge'), MODE = $('mode-badge'), YOLO = $('yolo-badge'), OLLAMA = $('ollama-badge'), INFER_BADGE = $('infer-badge'), CLOCK = $('clock');
  var INF_LIST = $('inf-list'), INF_COUNT = $('inf-count');
  var TTS_LIST = $('tts-list'), TTS_COUNT = $('tts-count');
  var VID_STALE = $('vid-stale'), FOOTER_IP = $('footer-ip');
  var SNAP_IMG = $('snap-img'), SNAP_EMPTY = $('snap-empty'), SNAP_AGE = $('snap-age'), SNAP_META = $('snap-meta');

  var failCount = 0;

  function init(){ 
    // 启动链式安全轮询
    fetchNow(); 
  }

  function ts(t){
    var d = new Date(t * 1000);
    return d.getHours().toString().padStart(2,'0') + ':' + 
           d.getMinutes().toString().padStart(2,'0') + ':' + 
           d.getSeconds().toString().padStart(2,'0');
  }

  function ago(t){
    if(!t) return '';
    var s = Math.max(0, (Date.now()/1000) - t);
    if(s < 60) return Math.round(s) + 's';
    if(s < 3600) return Math.round(s/60) + 'm';
    return Math.round(s/3600) + 'h';
  }

  function fmtMs(ms){
    if(!ms||ms===0) return '';
    if(ms >= 1000) return (ms/1000).toFixed(1) + 's';
    return ms + 'ms';
  }

  function fetchNow(){
    fetch('/api/dashboard')
      .then(function(r){ if(!r.ok) throw Error('HTTP '+r.status); return r.json(); })
      .then(function(d){
        failCount = 0;
        CONN.className = 'h-badge live'; CONN.textContent = '● ONLINE';
        render(d);
        // 成功获取后，等待 2 秒再进行下一次请求，避免高负载下堆积请求
        setTimeout(fetchNow, 2000);
      })
      .catch(function(e){
        failCount++;
        if(failCount >= 3){
          CONN.className = 'h-badge off'; CONN.textContent = '● OFFLINE';
          VID_STALE.style.display = 'inline';
        }
        // 发生失败时延长下一次尝试，为边缘端留出喘息时间
        setTimeout(fetchNow, 4000);
      });
  }

  function render(d){
    // Clock
    var now = new Date();
    CLOCK.textContent = now.getHours().toString().padStart(2,'0')+':'+
                        now.getMinutes().toString().padStart(2,'0')+':'+
                        now.getSeconds().toString().padStart(2,'0');

    // Footer IP 展示
    if(d.local_ip && FOOTER_IP.textContent !== d.local_ip){
      FOOTER_IP.textContent = d.local_ip;
    }

    // Mode badge
    var m = d.mode_name || '--';
    MODE.textContent = m;
    if(d.mode_idx === 1){
      MODE.style.background = 'rgba(218,54,51,.12)'; MODE.style.color = 'var(--red)';
    } else if(d.mode_idx >= 4){
      MODE.style.background = 'rgba(88,166,255,.08)'; MODE.style.color = 'var(--blue)';
    } else {
      MODE.style.background = 'var(--card)'; MODE.style.color = 'var(--muted)';
    }

    // YOLO badge
    if(d.yolo_enabled){
      var yr = d.yolo_running;
      YOLO.textContent = 'YOLO ' + (yr ? 'ON':'PAUSE');
      YOLO.style.background = yr ? 'rgba(46,160,67,.12)' : 'rgba(210,153,29,.12)';
      YOLO.style.color = yr ? 'var(--green)' : 'var(--yellow)';
    } else {
      YOLO.textContent = 'YOLO OFF';
      YOLO.style.background = 'var(--card)'; YOLO.style.color = 'var(--muted)';
    }

    // Ollama badge
    if(d.ollama_connected){
      OLLAMA.textContent = 'Ollama ON';
      OLLAMA.style.background = 'rgba(46,160,67,.12)';
      OLLAMA.style.color = 'var(--green)';
    } else {
      OLLAMA.textContent = 'Ollama OFF';
      OLLAMA.style.background = 'var(--card)'; OLLAMA.style.color = 'var(--muted)';
    }

    // Inference progress badge — 识别中脉冲指示器
    INFER_BADGE.style.display = (d.state === 'inferring') ? 'inline' : 'none';

    // Health badge — 综合判断系统健康度
    renderHealth(d);

    // Hardware bars
    setBar('gpu', d.gpu_percent || 0);
    setBar('cpu', d.cpu_percent || 0);
    setBar('mem', d.mem_percent || 0);

    var temp = parseFloat(d.temperature) || 0;
    var tEl = $('val-temp');
    if(temp > 0){
      tEl.textContent = temp.toFixed(0) + '°C';
      tEl.className = 'hw-temp ' + (temp >= 75 ? 'hot' : (temp >= 55 ? 'warm' : 'cool'));
    } else {
      tEl.textContent = '--°C';
      tEl.className = 'hw-temp cool';
    }

    // Status dots
    setDot('dot-yolo', d.yolo_running ? 'on' : (d.yolo_enabled ? 'paused' : 'off'));
    setDot('dot-depth', d.depth_camera_ok ? 'on' : 'off');
    setDot('dot-ollama', d.ollama_connected ? 'on' : 'off');
    setDot('dot-tts', d.tts_speaking ? 'on' : 'off');

    // Inference logs (reverse: newest first)
    var infs = d.inf_logs || [];
    INF_COUNT.textContent = infs.length;
    if(infs.length === 0){
      INF_LIST.innerHTML = '<div class="log-empty">暂无识别记录</div>';
    } else {
      var h = '';
      for(var i = infs.length-1; i >= 0; i--){
        var e = infs[i];
        var cls = e.success ? 'inf-ok' : 'inf-fail';
        var icon = e.success ? '✓' : '✗';
        var txt = e.text || '(空)';
        h += '<div class="log-entry '+cls+'">'+
               '<span class="log-time">'+ts(e.time)+'</span>'+
               '<span>'+icon+'</span>'+
               '<span class="log-text">'+esc(txt)+'</span>'+
               '<span class="log-lat">'+fmtMs(e.latency_ms)+'</span>'+
             '</div>';
      }
      INF_LIST.innerHTML = h;
    }

    // TTS logs
    var tts = d.tts_logs || [];
    TTS_COUNT.textContent = tts.length;
    if(tts.length === 0){
      TTS_LIST.innerHTML = '<div class="log-empty">暂无播报记录</div>';
    } else {
      var h2 = '';
      for(var j = tts.length-1; j >= 0; j--){
        var t = tts[j];
        var p = t.priority != null ? t.priority : 3;
        var cls2 = 'tts-p' + p;
        h2 += '<div class="log-entry '+cls2+'">'+
                '<span class="log-time">'+ts(t.time)+'</span>'+
                '<span class="log-prio">P'+p+'</span>'+
                '<span class="log-text">'+esc(t.text||'(空)')+'</span>'+
                '<span class="log-engine">'+(t.engine||'')+'</span>'+
              '</div>';
      }
      TTS_LIST.innerHTML = h2;
    }

    // Snapshot
    renderSnap(d);
  }

  function setBar(id, pct){
    var raw = parseFloat(pct);
    var p = Number.isFinite(raw) ? Math.min(100, Math.max(0, raw)) : 0;
    var barId = id.startsWith('bar') ? id : 'bar-' + id;
    var valId = id.startsWith('val') ? id : 'val-' + id;
    var bar = $(barId);
    var val = $(valId);
    if(bar) bar.style.width = p + '%';
    if(val) val.textContent = p.toFixed(0) + '%';
  }

  function renderHealth(d){
    var issues = [];
    // 核心服务
    if(!d.ollama_connected) issues.push('Ollama 离线');
    // 感知服务
    if(d.yolo_enabled && !d.yolo_running && d.mode_idx === 1) issues.push('YOLO 暂停');
    if(!d.depth_camera_ok) issues.push('深度相机异常');
    // 近 60 秒内有 P0 危险
    var now = Date.now()/1000;
    if(d.last_detection_priority === 0 && (now - d.last_detection_time) < 60){
      issues.push('检测到危险障碍');
    }

    if(issues.length === 0){
      HEALTH.textContent = '系统正常';
      HEALTH.className = 'h-badge health-ok';
    } else if(issues.length === 1 && !d.ollama_connected){
      // 唯一问题是 Ollama 离线 → 严重
      HEALTH.textContent = '推理离线';
      HEALTH.className = 'h-badge health-crit';
    } else if(issues.length === 1){
      HEALTH.textContent = issues[0];
      HEALTH.className = 'h-badge health-warn';
    } else {
      HEALTH.textContent = issues[0] + ' +' + (issues.length-1);
      HEALTH.className = 'h-badge health-crit';
    }
  }

  function setDot(id, state){
    var el = $(id);
    if(el) el.className = 's-dot ' + state;
  }

  function esc(s){
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function renderSnap(d){
    if(!d.last_snapshot_url){
      SNAP_IMG.style.display = 'none';
      SNAP_META.style.display = 'none';
      SNAP_EMPTY.style.display = 'block';
      SNAP_AGE.textContent = '暂无快照';
      return;
    }

    // 只有 URL 变化时才重新加载图片（省带宽）
    if(SNAP_IMG.dataset.url !== d.last_snapshot_url){
      SNAP_IMG.src = d.last_snapshot_url + '?t=' + Date.now();
      SNAP_IMG.dataset.url = d.last_snapshot_url;
    }

    SNAP_IMG.style.display = 'block';
    SNAP_EMPTY.style.display = 'none';

    // 时间标记
    var ageStr = d.last_snapshot_time ? (ago(d.last_snapshot_time) + '前') : '--';
    var modeIdx = d.last_snapshot_mode || 0;
    var modeNames = ['','障碍物检测','文字识别','人脸检测','场景描述','图文问答'];
    var modeName = modeNames[modeIdx] || '模式'+modeIdx;
    var modeClasses = ['','det','recog','face','scene','qa'];
    var modeCls = modeClasses[modeIdx] || '';

    SNAP_AGE.innerHTML = '模式' + modeIdx + ' · ' + ageStr;
    SNAP_META.style.display = 'flex';
    SNAP_META.innerHTML = '<span>'+modeName+'</span>'+
      '<span class="snap-mode '+modeCls+'">拍摄于 ' + ts(d.last_snapshot_time) + '</span>';
  }

  init();
})();
</script>
</body>
</html>
"""


def register_dashboard(app):
    """
    在现有 Flask app 上注册 / 和 /api/dashboard 路由。

    Args:
        app: Flask 实例
    """
    from src.dashboard_status import system_status

    @app.route("/")
    def dashboard_page():
        # 移除 render_template_string 的 CPU 开销，直接返回静态 HTML
        return Response(_DASHBOARD_HTML, mimetype="text/html")

    @app.route("/api/dashboard")
    def dashboard_api():
        try:
            # 获取系统监控状态字典
            status_data = system_status.snapshot()
            # 动态混入当前的局域网 IP 供前端渲染
            status_data["local_ip"] = _LOCAL_IP
            return jsonify(status_data)
        except Exception as e:
            logger.warning(f"仪表板快照异常: {e}")
            return jsonify({"error": "状态不可用", "local_ip": _LOCAL_IP})

    logger.info(f"遥测仪表板已挂载: / (本地调试地址: http://{_LOCAL_IP}:5000)")