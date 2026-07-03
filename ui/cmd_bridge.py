#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI内嵌命令桥 (cmd_bridge) — 扩展版
====================================
在GUI进程内起一个 127.0.0.1 本地socket命令服务, 让外部(经SSH本地)可安全操控点锡机。
指令经 pyqtSignal 投递到Qt主线程执行, 与UI共享同一 _motor 对象和串口/相机。
仅绑127.0.0.1, 外网不可达; 无鉴权(依赖localhost + SSH双重边界)。

协议(JSON单条, \n可选)。用 "op" 字段分流, 缺省(无op但有cmd)向后兼容为运动指令:

  运动(即发即回, 不等完成):
    {"cmd":1,"args":[500,500]}                → {"ok":true}
    {"op":"cmd","cmd":7,"args":[1]}           → {"ok":true}

  运动+等完成(下发并阻塞到 is_busy()=False, 运动完成上行反馈核心):
    {"op":"cmd_wait","cmd":1,"args":[500,500],"timeout":30}
        → {"ok":true,"done":true,"x":500.0,"y":500.0,"z":100,"state":0,"busy":false}

  查询实时坐标/状态(worker线程直读, motor方法自带锁, 快):
    {"op":"query"}
        → {"ok":true,"x":..,"y":..,"z":..,"state":0,"busy":false,"online":true,"laser":..}

  操作UI控件(调 MainWindow 任意方法, 最高权限动态派发, 主线程执行):
    {"op":"ui","method":"_cmd_home","args":[]}       → {"ok":true,"result":...}
    {"op":"ui","method":"switch_mode","args":["solder"]}

  抓取顶部相机帧存jpeg(主线程, cv2):
    {"op":"grab","path":"/tmp/ai_frame.jpg"}
        → {"ok":true,"path":"/tmp/ai_frame.jpg","w":1920,"h":1080}
"""
import socket, json, threading, time
from PyQt5.QtCore import QObject, pyqtSignal

CMD_HOST = "127.0.0.1"
CMD_PORT = 8931


class CmdBridge(QObject):
    _invoke = pyqtSignal(object)   # 从worker线程emit -> 主线程排队执行

    def __init__(self, win):
        super().__init__()
        self.win = win
        self._invoke.connect(self._on_invoke)   # AutoConnection: 跨线程emit自动QueuedConnection

    # ---------- 主线程执行体 ----------
    def _on_invoke(self, job):
        """在Qt主线程执行需要碰UI/串口/相机的操作。"""
        op = job.get("op", "cmd")
        try:
            if op in ("cmd", "cmd_wait"):
                cmd = int(job["cmd"]); args = job.get("args", []) or []
                r = self.win._send_cmd(cmd, *args)
                job["result"] = {"ok": bool(r)}
            elif op == "ui":
                method = job["method"]; args = job.get("args", []) or []
                fn = getattr(self.win, method, None)
                if fn is None or not callable(fn):
                    job["result"] = {"ok": False, "error": "no such method: %s" % method}
                else:
                    ret = fn(*args)
                    # 返回值尽量JSON化, 不行就转str
                    try:
                        json.dumps(ret)
                        rv = ret
                    except Exception:
                        rv = repr(ret)
                    job["result"] = {"ok": True, "result": rv}
            elif op == "grab":
                job["result"] = self._do_grab(job.get("path") or "/tmp/ai_frame.jpg")
            else:
                job["result"] = {"ok": False, "error": "unknown op: %s" % op}
        except Exception as e:
            job["result"] = {"ok": False, "error": str(e)}
        finally:
            job["event"].set()

    def _do_grab(self, path):
        """主线程抓顶部相机单帧存jpeg。复用 MainWindow._grab_top_frame。"""
        try:
            import cv2
        except Exception as e:
            return {"ok": False, "error": "cv2 import: %s" % e}
        grab = getattr(self.win, "_grab_top_frame", None)
        if grab is None:
            return {"ok": False, "error": "no _grab_top_frame"}
        frame = grab()
        if frame is None:
            return {"ok": False, "error": "grab returned None"}
        try:
            ok = cv2.imwrite(path, frame)
            if not ok:
                return {"ok": False, "error": "imwrite failed: %s" % path}
            h, w = frame.shape[:2]
            return {"ok": True, "path": path, "w": int(w), "h": int(h)}
        except Exception as e:
            return {"ok": False, "error": "imwrite: %s" % e}

    # ---------- worker线程直读(motor方法自带_lock, 线程安全) ----------
    def _do_query(self):
        m = getattr(self.win, "_motor", None)
        if m is None:
            return {"ok": False, "error": "motor not connected"}
        try:
            x, y = m.get_position()
            z = m.get_z() if hasattr(m, "get_z") else None
            st = m.get_state() if hasattr(m, "get_state") else None
            busy = m.is_busy() if hasattr(m, "is_busy") else None
            online = m.is_online() if hasattr(m, "is_online") else None
            laser = m.get_laser() if hasattr(m, "get_laser") else None
            return {"ok": True, "x": x, "y": y, "z": z, "state": st,
                    "busy": bool(busy), "online": bool(online), "laser": laser}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _wait_done(self, timeout):
        """两阶段轮询运动完成: 先等is_busy变True(确认已启动),再等变False(完成)。
        返回完成后的坐标/状态快照。
        修复: STM32状态帧~100ms才上报, 下发后立即读is_busy会拿到运动前的旧值(False),
        导致秒过。故必须先确认busy置起。"""
        m = getattr(self.win, "_motor", None)
        if m is None:
            return {"done": False, "error": "motor not connected"}

        def _busy():
            try:
                return bool(m.is_busy())
            except Exception:
                return False

        t0 = time.time()
        # 阶段1: 等待运动启动(is_busy变True)。窗口取min(2s, timeout)。
        start_win = min(2.0, timeout)
        started = False
        while time.time() - t0 < start_win:
            if _busy():
                started = True
                break
            time.sleep(0.03)

        # 阶段2: 若已启动, 等待is_busy变False(运动完成)。
        # 若start_win内从未见busy=True: 可能是极短运动已完成, 直接读快照。
        if started:
            while time.time() - t0 < timeout:
                if not _busy():
                    break
                time.sleep(0.03)

        done = not _busy()
        snap = self._do_query()
        snap["done"] = done
        snap["started"] = started
        if not done:
            snap["error"] = snap.get("error", "wait timeout")
        return snap

    # ---------- socket线程入口: 路由 ----------
    def handle(self, data):
        op = data.get("op", "cmd")

        # 纯读, 不进主线程
        if op == "query":
            r = self._do_query()
            return r

        # 需主线程: cmd / cmd_wait / ui / grab
        job = {"op": op, "cmd": data.get("cmd"), "args": data.get("args", []),
               "method": data.get("method"), "path": data.get("path"),
               "event": threading.Event()}
        self._invoke.emit(job)
        wait_t = 20 if op == "grab" else 15
        if not job["event"].wait(timeout=wait_t):
            return {"ok": False, "error": "main-thread timeout"}
        res = job.get("result", {"ok": False, "error": "no result"})

        # cmd_wait: 主线程已下发成功, 再在socket线程轮询完成(不阻塞主线程/UI)
        if op == "cmd_wait" and res.get("ok"):
            timeout = float(data.get("timeout", 30))
            wd = self._wait_done(timeout)
            res.update(wd)
        return res


def _serve(bridge):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((CMD_HOST, CMD_PORT)); srv.listen(5)
    except Exception as e:
        print("[cmd_bridge] bind failed:", e); return
    print("[cmd_bridge] listening on %s:%d (op: cmd/cmd_wait/query/ui/grab)" % (CMD_HOST, CMD_PORT))
    while True:
        try:
            conn, _ = srv.accept()
        except Exception:
            continue
        try:
            conn.settimeout(5.0)
            chunks = []
            # 读到一个完整JSON(遇\n或parse成功即停), 兼容大payload
            buf = b""
            while True:
                d = conn.recv(8192)
                if not d:
                    break
                buf += d
                if b"\n" in d or len(buf) > 65536:
                    break
                # 尝试提前解析(无换行的单包)
                try:
                    json.loads(buf.decode("utf-8", "replace").strip())
                    break
                except Exception:
                    continue
            data = json.loads(buf.decode("utf-8", "replace").strip())
            res = bridge.handle(data)
        except Exception as e:
            res = {"ok": False, "error": "parse/exec: %s" % e}
        try:
            conn.sendall((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception:
            pass
        finally:
            try: conn.close()
            except Exception: pass


def attach_cmd_bridge(win):
    bridge = CmdBridge(win)
    win._cmd_bridge = bridge   # 保引用防GC
    t = threading.Thread(target=_serve, args=(bridge,), daemon=True)
    t.start()
    return bridge
