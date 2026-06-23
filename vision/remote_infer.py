#!/usr/bin/env python3
"""
外部网口推理客户端 (RK3588 → Windows 11 推理服务)
=====================================================
协议: REST + 高质量JPEG
  健康检查: GET  {base_url}/health   → 200 {"status":"ok"}
  推理:     POST {base_url}/infer    Content-Type: image/jpeg
            body: JPEG字节 (quality=95)
            返回: {"detections":[{"bbox":[x1,y1,x2,y2],"score":0.93,"class":0}, ...],
                   "elapsed_ms":45}
类别顺序与本地一致: 0=pad, 1=hole, 2=qfn
返回格式与本地 infer() 完全一致: (bboxes[N,4] np, scores[N] np, class_ids[N] np-int)
坐标系: 与发送的ROI图像像素坐标一致 (上层反算逻辑本地/远程通用)
"""

import cv2
import numpy as np
import requests

# Windows 11 推理服务地址 (硬编码)
DEFAULT_BASE_URL = "http://192.168.137.222:8000"
JPEG_QUALITY = 95


class RemoteInferClient:
    """外部推理服务HTTP客户端"""

    def __init__(self, base_url=DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self._sess = requests.Session()

    def check_health(self, timeout=0.8):
        """健康检查。返回 True/False，不抛异常。"""
        try:
            r = self._sess.get(self.base_url + "/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def infer(self, roi_bgr, conf_thresh=None, timeout=5.0):
        """远程推理。roi_bgr: 正方形BGR ROI。
        返回 (bboxes[N,4], scores[N], class_ids[N]) numpy, 与本地infer()格式一致。
        网络/解析失败抛异常, 由调用方捕获处理。"""
        ok, buf = cv2.imencode('.jpg', roi_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG编码失败")
        headers = {"Content-Type": "image/jpeg"}
        params = {}
        if conf_thresh is not None:
            params["conf"] = float(conf_thresh)
        r = self._sess.post(self.base_url + "/infer", data=buf.tobytes(),
                            headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        dets = data.get("detections", [])
        if not dets:
            return (np.empty((0, 4)), np.empty((0,)),
                    np.empty((0,), dtype=int))
        bboxes = np.array([d["bbox"] for d in dets], dtype=np.float64)
        scores = np.array([d.get("score", 1.0) for d in dets], dtype=np.float64)
        class_ids = np.array([int(d.get("class", 0)) for d in dets], dtype=int)
        return bboxes, scores, class_ids
