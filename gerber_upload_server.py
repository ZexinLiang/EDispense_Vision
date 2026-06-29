#!/usr/bin/env python3
"""
Gerber 无线上传服务 (点锡机 RK3588)
====================================
手机/电脑浏览器打开 http://<板子IP>:8090 即可拖拽上传 gerber zip,
文件固定存入 SD 卡指定文件夹。纯标准库实现(无需Flask), 板子直接能跑。

部署:
    python3 gerber_upload_server.py
    然后浏览器访问 http://192.168.137.232:8090
"""
import os
import re
import io
import json
import time
import html
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ============ 配置区 ============
# SD卡上传目录 (待连板确认: SD卡通常挂在 /media/elf/OPI_BOOT)
UPLOAD_DIR = os.environ.get('GERBER_UPLOAD_DIR', '/media/elf/OPI_BOOT/Gerber')
PORT = int(os.environ.get('GERBER_UPLOAD_PORT', '8090'))
ALLOWED_EXT = {'.zip', '.rar', '.7z', '.gerber'}
MAX_SIZE = 50 * 1024 * 1024  # 单文件上限 50MB
# ================================

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Gerber 上传 · 点锡机</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  body {
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: linear-gradient(160deg, #1c1c1e 0%, #2c2c2e 100%);
    color: #f2f2f7; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 24px 16px;
  }
  .wrap { width: 100%; max-width: 560px; }
  h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .sub { font-size: 13px; color: #98989f; margin-bottom: 22px; }
  .card {
    background: rgba(44,44,46,.85); border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px; padding: 20px; margin-bottom: 16px;
    backdrop-filter: blur(10px);
  }
  #drop {
    border: 2px dashed rgba(255,255,255,.18); border-radius: 14px;
    padding: 38px 16px; text-align: center; cursor: pointer;
    transition: all .2s; background: rgba(255,255,255,.02);
  }
  #drop.hover { border-color: #0a84ff; background: rgba(10,132,255,.10); }
  #drop .ic { font-size: 44px; margin-bottom: 10px; }
  #drop .t1 { font-size: 16px; font-weight: 600; }
  #drop .t2 { font-size: 12px; color: #98989f; margin-top: 6px; }
  input[type=file] { display: none; }
  .bar { height: 6px; background: rgba(255,255,255,.1); border-radius: 3px; overflow: hidden; margin-top: 16px; display: none; }
  .bar > i { display: block; height: 100%; width: 0; background: linear-gradient(90deg,#0a84ff,#30d158); transition: width .2s; }
  .msg { font-size: 13px; margin-top: 12px; min-height: 18px; }
  .ok { color: #30d158; } .err { color: #ff453a; }
  .files-title { font-size: 14px; font-weight: 600; margin-bottom: 10px; color: #c7c7cc; }
  .frow { display: flex; justify-content: space-between; align-items: center;
          padding: 10px 12px; border-radius: 10px; background: rgba(255,255,255,.04); margin-bottom: 6px; font-size: 13px; }
  .frow .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: 10px; }
  .frow .sz { color: #98989f; flex-shrink: 0; }
  .empty { color: #6a6a6e; font-size: 13px; text-align: center; padding: 12px; }
  .path { font-size: 11px; color: #6a6a6e; margin-top: 14px; word-break: break-all; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Gerber 上传</h1>
  <div class="sub">点锡机 · 拖拽或点击选择 gerber 压缩包</div>

  <div class="card">
    <div id="drop">
      <div class="ic">📁</div>
      <div class="t1">点击选择 或 拖拽文件到此</div>
      <div class="t2">支持 .zip / .rar / .7z，最大 50MB</div>
    </div>
    <input type="file" id="file" accept=".zip,.rar,.7z,.gerber">
    <div class="bar"><i></i></div>
    <div class="msg" id="msg"></div>
    <div class="path" id="path"></div>
  </div>

  <div class="card">
    <div class="files-title">已上传文件</div>
    <div id="files"><div class="empty">加载中…</div></div>
  </div>
</div>
<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const bar = document.querySelector('.bar');
const barI = document.querySelector('.bar > i');
const msg = document.getElementById('msg');
const filesBox = document.getElementById('files');

drop.onclick = () => file.click();
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('hover'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('hover'); }));
drop.addEventListener('drop', ev => { if (ev.dataTransfer.files.length) upload(ev.dataTransfer.files[0]); });
file.onchange = () => { if (file.files.length) upload(file.files[0]); };

function fmt(b){ return b<1024?b+' B':b<1048576?(b/1024).toFixed(1)+' KB':(b/1048576).toFixed(1)+' MB'; }

function upload(f){
  msg.className='msg'; msg.textContent='';
  bar.style.display='block'; barI.style.width='0';
  const fd = new FormData(); fd.append('file', f);
  const xhr = new XMLHttpRequest();
  xhr.open('POST','/upload');
  xhr.upload.onprogress = e => { if(e.lengthComputable) barI.style.width=(e.loaded/e.total*100)+'%'; };
  xhr.onload = () => {
    let r={}; try{ r=JSON.parse(xhr.responseText);}catch(e){}
    if(xhr.status===200 && r.ok){ msg.className='msg ok'; msg.textContent='✓ 上传成功: '+r.name; loadFiles(); }
    else { msg.className='msg err'; msg.textContent='✗ '+(r.error||'上传失败'); }
    setTimeout(()=>bar.style.display='none',600);
  };
  xhr.onerror = () => { msg.className='msg err'; msg.textContent='✗ 网络错误'; };
  xhr.send(fd);
}

function loadFiles(){
  fetch('/list').then(r=>r.json()).then(d=>{
    document.getElementById('path').textContent='存储目录: '+d.dir;
    if(!d.files.length){ filesBox.innerHTML='<div class="empty">暂无文件</div>'; return; }
    filesBox.innerHTML = d.files.map(f=>
      `<div class="frow"><span class="nm">${f.name}</span><span class="sz">${fmt(f.size)}</span></div>`).join('');
  }).catch(()=>{ filesBox.innerHTML='<div class="empty err">列表加载失败</div>'; });
}
loadFiles();
</script>
</body>
</html>"""


def _parse_multipart_file(rfile, boundary, clen):
    """不依赖cgi: 从body中解析第一个文件字段, 返回(filename, data)或(None,None)。"""
    data = rfile.read(clen)
    bnd = ('--' + boundary).encode()
    parts = data.split(bnd)
    for part in parts:
        if b'Content-Disposition' not in part:
            continue
        # 分离头部和内容(\r\n\r\n)
        idx = part.find(b'\r\n\r\n')
        if idx < 0:
            continue
        head = part[:idx].decode('utf-8', 'ignore')
        if 'filename=' not in head:
            continue
        m = re.search(r'filename="([^"]*)"', head)
        fname = m.group(1) if m else ''
        body = part[idx + 4:]
        # 去掉结尾的\r\n
        if body.endswith(b'\r\n'):
            body = body[:-2]
        return fname, body
    return None, None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype + ('; charset=utf-8' if 'json' in ctype or 'html' in ctype else ''))
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass  # 静默

    def do_GET(self):
        if self.path == '/' or self.path.startswith('/index'):
            self._send(200, PAGE, 'text/html')
        elif self.path == '/list':
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            files = []
            for n in sorted(os.listdir(UPLOAD_DIR)):
                p = os.path.join(UPLOAD_DIR, n)
                if os.path.isfile(p):
                    files.append({'name': n, 'size': os.path.getsize(p)})
            self._send(200, json.dumps({'dir': UPLOAD_DIR, 'files': files}))
        else:
            self._send(404, json.dumps({'error': 'not found'}))

    def do_POST(self):
        if self.path != '/upload':
            self._send(404, json.dumps({'error': 'not found'}))
            return
        try:
            clen = int(self.headers.get('Content-Length', 0))
            if clen > MAX_SIZE:
                self._send(413, json.dumps({'ok': False, 'error': '文件超过50MB上限'}))
                return
            ctype = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in ctype:
                self._send(400, json.dumps({'ok': False, 'error': '格式错误'}))
                return
            m = re.search(r'boundary=([^;]+)', ctype)
            if not m:
                self._send(400, json.dumps({'ok': False, 'error': '缺少boundary'}))
                return
            boundary = m.group(1).strip().strip('"')
            raw_name, data = _parse_multipart_file(self.rfile, boundary, clen)
            if not raw_name or data is None:
                self._send(400, json.dumps({'ok': False, 'error': '无文件'}))
                return
            raw_name = os.path.basename(raw_name)
            ext = os.path.splitext(raw_name)[1].lower()
            if ext not in ALLOWED_EXT:
                self._send(400, json.dumps({'ok': False, 'error': f'不支持的类型 {ext}'}))
                return
            # 安全文件名: 仅保留字母数字._-中文
            safe = re.sub(r'[^\w.\-\u4e00-\u9fff]', '_', raw_name)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            dst = os.path.join(UPLOAD_DIR, safe)
            with open(dst, 'wb') as f:
                f.write(data)
            self._send(200, json.dumps({'ok': True, 'name': safe, 'size': len(data)}))
        except Exception as e:
            self._send(500, json.dumps({'ok': False, 'error': str(e)}))


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return '0.0.0.0'


def main():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    srv = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    print(f"Gerber上传服务已启动")
    print(f"  访问: http://{_lan_ip()}:{PORT}")
    print(f"  存储: {UPLOAD_DIR}")
    srv.serve_forever()


if __name__ == '__main__':
    main()
