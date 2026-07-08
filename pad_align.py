#!/usr/bin/env python3
"""
Gerber 焊盘 ↔ 相机检测焊盘 自动配准
=====================================
求相似变换 (scale s, rotation theta, translation tx,ty), 使 gerber 焊盘(mm)
投影到相机全帧像素坐标, 与 YOLO 检测到的 pad 中心点云对齐。

坐标约定 (与 solder_ui._tpl_to_frame_px 完全一致):
    xs =  x_mm * s
    ys = -y_mm * s
    u  = cos(th)*xs - sin(th)*ys + tx
    v  = sin(th)*xs + cos(th)*ys + ty
即先按 s 缩放 + y 翻转, 再旋转 th, 再平移。返回的 (s, theta, tx, ty)
可直接填入 UI 的 _tpl_s / _tpl_theta / _tpl_tx / _tpl_ty。

算法: 受约束 2 点 RANSAC (scale/rotation 先验作硬过滤 + 最近邻内点计数)
      → ICP (Umeyama 相似变换) 精修。
适应: 部分入镜(板子仅部分焊盘可见)、未知对应、无 Mark 点。
"""
import numpy as np


def gerber_to_tpl_xy(pts_mm):
    """gerber (x_mm,y_mm) -> (xs,ys) 未旋转未平移的模板坐标 (y翻转)。s=1。"""
    out = np.empty_like(pts_mm, dtype=float)
    out[:, 0] = pts_mm[:, 0]
    out[:, 1] = -pts_mm[:, 1]
    return out


def apply_sim(pts, s, theta, tx, ty):
    """对 (N,2) 点应用 相似变换。pts 已是 y翻转后的模板坐标(xs,ys @ s=1)。"""
    c, sn = np.cos(theta), np.sin(theta)
    xs = pts[:, 0] * s
    ys = pts[:, 1] * s
    u = c * xs - sn * ys + tx
    v = sn * xs + c * ys + ty
    return np.stack([u, v], axis=1)


def _nn_inliers(src, dst, tol):
    """src 每点到 dst 最近距离, 返回 (inlier_mask, dists, nn_idx)。"""
    # 分块算距离矩阵防内存爆
    nn_idx = np.empty(len(src), dtype=int)
    dists = np.empty(len(src), dtype=float)
    B = 512
    for i in range(0, len(src), B):
        sub = src[i:i + B]
        d2 = ((sub[:, None, 0] - dst[None, :, 0]) ** 2 +
              (sub[:, None, 1] - dst[None, :, 1]) ** 2)
        j = np.argmin(d2, axis=1)
        nn_idx[i:i + B] = j
        dists[i:i + B] = np.sqrt(d2[np.arange(len(sub)), j])
    return dists <= tol, dists, nn_idx


def umeyama_sim(src, dst):
    """求 src->dst 的相似变换 (s,theta,tx,ty)。src,dst: (N,2) 对应点。"""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    ss = src - mu_s
    dd = dst - mu_d
    cov = dd.T @ ss / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    var_s = (ss ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    theta = np.arctan2(R[1, 0], R[0, 0])
    return s, theta, t[0], t[1]


def register(gerber_mm, det_px, scale_lo, scale_hi,
             theta_max_deg=20.0, tol_px=18.0,
             iters=20000, min_inliers=25, seed=0):
    """
    gerber_mm: (Ng,2) gerber 焊盘 mm 坐标
    det_px:    (Nd,2) 检测焊盘 全帧像素坐标
    scale_lo/hi: 允许的 s (px/mm) 范围
    返回 dict: s,theta,tx,ty, inliers, rmse, ok
    """
    rng = np.random.default_rng(seed)
    G = gerber_to_tpl_xy(np.asarray(gerber_mm, float))  # y翻转后, s=1
    D = np.asarray(det_px, float)
    Ng, Nd = len(G), len(D)
    th_max = np.radians(theta_max_deg)

    best = None
    gi = rng.integers(0, Ng, size=(iters, 2))
    di = rng.integers(0, Nd, size=(iters, 2))
    for k in range(iters):
        a, b = gi[k]
        p, q = di[k]
        if a == b or p == q:
            continue
        g1, g2 = G[a], G[b]
        d1, d2 = D[p], D[q]
        vg = g2 - g1
        vd = d2 - d1
        lg = np.hypot(*vg)
        ld = np.hypot(*vd)
        if lg < 3 or ld < 3:
            continue
        s = ld / lg
        if s < scale_lo or s > scale_hi:
            continue
        theta = np.arctan2(vd[1], vd[0]) - np.arctan2(vg[1], vg[0])
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        if abs(theta) > th_max:
            continue
        # translation from g1->d1
        c, sn = np.cos(theta), np.sin(theta)
        u1 = c * g1[0] * s - sn * g1[1] * s
        v1 = sn * g1[0] * s + c * g1[1] * s
        tx = d1[0] - u1
        ty = d1[1] - v1
        proj = apply_sim(G, s, theta, tx, ty)
        mask, dists, nn = _nn_inliers(proj, D, tol_px)
        # 按"唯一被覆盖的检测点数"评分, 惩罚多gerber坍缩到同一det的退化解
        score = int(len(np.unique(nn[mask]))) if mask.any() else 0
        if best is None or score > best[0]:
            best = (score, s, theta, tx, ty)

    if best is None or best[0] < min_inliers:
        return {"ok": False, "inliers": 0 if best is None else best[0]}

    _, s, theta, tx, ty = best
    # ICP 精修
    for _ in range(15):
        proj = apply_sim(G, s, theta, tx, ty)
        mask, dists, nn = _nn_inliers(proj, D, tol_px)
        if mask.sum() < 6:
            break
        src = G[mask]
        dst = D[nn[mask]]
        # 用当前 s 翻转坐标已含在 G, umeyama 求 G->D
        s2, th2, tx2, ty2 = umeyama_sim(src, dst)
        if (abs(s2 - s) < 1e-4 and abs(th2 - theta) < 1e-5 and
                abs(tx2 - tx) < 1e-2 and abs(ty2 - ty) < 1e-2):
            s, theta, tx, ty = s2, th2, tx2, ty2
            break
        s, theta, tx, ty = s2, th2, tx2, ty2

    proj = apply_sim(G, s, theta, tx, ty)
    mask, dists, _ = _nn_inliers(proj, D, tol_px)
    rmse = float(np.sqrt((dists[mask] ** 2).mean())) if mask.any() else 1e9
    return {"ok": True, "s": float(s), "theta": float(theta),
            "tx": float(tx), "ty": float(ty),
            "inliers": int(mask.sum()), "rmse": rmse, "theta_deg": float(np.degrees(theta))}


if __name__ == "__main__":
    import sys, os, json
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gerber_paste_parser as gp
    res = gp.extract_paste_targets('assets/1.zip')
    gerber = np.array([[p['x'], p['y']] for p in res['pads']])
    d = json.load(open('assets/pad_dets.json'))
    det = np.array([[x[0], x[1]] for x in d['dets'] if x[5] == 0])
    # 尺度先验: 板框宽 ~95mm, ROI 1080px, 板占~70% => s ~ 5..14 px/mm
    r = register(gerber, det, scale_lo=4.0, scale_hi=16.0,
                 theta_max_deg=20, tol_px=18, iters=30000)
    print(json.dumps(r, indent=2))
