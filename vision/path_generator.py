"""
点锡路径生成器
输入：YOLO检测结果 (bboxes, scores, class_ids)
输出：点锡路径坐标序列 + 可选G-code

策略：
1. 小焊盘(面积<阈值) → 中心单点
2. 大焊盘(面积>=阈值) → 网格多点填充
3. 路径优化：贪心最近邻(减少空行程)
4. 坐标系：像素坐标，预留仿射变换接口
"""

import numpy as np
import json
import cv2

# ===== 参数配置 =====
# 大焊盘面积阈值(像素^2)，超过此值用多点填充
LARGE_PAD_AREA = 2000
# 多点填充时点间距(像素)
FILL_SPACING = 15
# 点锡停留时间(ms)
DWELL_TIME_MS = 200
# 移动速度(mm/s) - 用于G-code
MOVE_SPEED = 50
# 点锡高度(mm) - Z轴
Z_DISPENSE = 0.5
Z_TRAVEL = 5.0
# 像素→物理坐标仿射变换矩阵 (2x3)，默认单位矩阵(像素=物理)
# 标定后替换为实际值: [[sx, 0, tx], [0, sy, ty]]
AFFINE_MATRIX = np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float64)


def pixel_to_physical(points):
    """像素坐标转物理坐标（通过仿射变换）"""
    if len(points) == 0:
        return []
    pts = np.array(points, dtype=np.float64)
    # 齐次坐标
    ones = np.ones((len(pts), 1))
    pts_h = np.hstack([pts, ones])
    physical = (AFFINE_MATRIX @ pts_h.T).T
    return physical.tolist()


def generate_dispense_points(bboxes, scores, class_ids):
    """
    根据检测结果生成点锡坐标
    
    Args:
        bboxes: [[x1,y1,x2,y2], ...] 原图像素坐标
        scores: [float, ...]
        class_ids: [int, ...]
    
    Returns:
        list of dict: [{"x":px,"y":py,"type":"single"/"fill","class":id,"dwell":ms,
                        "pad":焊盘索引,"pad_cx":焊盘中心x,"pad_cy":焊盘中心y}, ...]
    """
    all_points = []
    
    for pad_idx, (bbox, score, cls_id) in enumerate(zip(bboxes, scores, class_ids)):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        pad_meta = {"pad": pad_idx, "pad_cx": float(cx), "pad_cy": float(cy),
                    "pad_w": float(w), "pad_h": float(h)}
        
        if area >= LARGE_PAD_AREA:
            # 大焊盘：网格填充
            margin = FILL_SPACING * 0.5
            xs = np.arange(x1 + margin, x2 - margin + 1, FILL_SPACING)
            ys = np.arange(y1 + margin, y2 - margin + 1, FILL_SPACING)
            if len(xs) == 0:
                xs = [cx]
            if len(ys) == 0:
                ys = [cy]
            for iy, y in enumerate(ys):
                # 蛇形走位减少空行程
                row_xs = xs if iy % 2 == 0 else xs[::-1]
                for x in row_xs:
                    all_points.append({
                        "x": float(x), "y": float(y),
                        "type": "fill", "class": int(cls_id),
                        "score": float(score), "dwell": DWELL_TIME_MS,
                        **pad_meta
                    })
        else:
            # 小焊盘：中心单点
            all_points.append({
                "x": float(cx), "y": float(cy),
                "type": "single", "class": int(cls_id),
                "score": float(score), "dwell": DWELL_TIME_MS,
                **pad_meta
            })
    
    return all_points


def optimize_path(points):
    """
    路径排序：按焊盘分组，焊盘间做行带蛇形扫描(逐行往返),焊盘内点保持原顺序。
    避免全局最近邻把同一焊盘的点打散导致喷头在焊盘间反复横跳。
    """
    if len(points) <= 1:
        return points

    # 1. 按焊盘分组(保持组内原顺序)
    groups = {}
    order = []
    for p in points:
        pid = p.get("pad", 0)
        if pid not in groups:
            groups[pid] = []
            order.append(pid)
        groups[pid].append(p)

    # 2. 每个焊盘取中心代表点 + 焊盘高度
    pads = []
    for pid in order:
        g = groups[pid]
        cx = g[0].get("pad_cx", float(np.mean([q["x"] for q in g])))
        cy = g[0].get("pad_cy", float(np.mean([q["y"] for q in g])))
        ph = g[0].get("pad_h", 0.0)
        pads.append({"pid": pid, "cx": cx, "cy": cy, "h": ph})

    if len(pads) == 1:
        return groups[pads[0]["pid"]]

    # 3. 按y排序后聚类成行: 与当前行基准y之差超过阈值(约半个焊盘高度,兜底用中位y间距)即换行
    pads_by_y = sorted(pads, key=lambda d: d["cy"])
    med_h = float(np.median([d["h"] for d in pads_by_y if d["h"] > 0])) if any(d["h"] > 0 for d in pads_by_y) else 0.0
    if med_h <= 0:
        ys = [d["cy"] for d in pads_by_y]
        diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a > 1e-3]
        med_h = float(np.median(diffs)) if diffs else 1.0
    row_tol = max(med_h * 0.6, 1.0)

    rows = []
    cur_row = [pads_by_y[0]]
    row_ref_y = pads_by_y[0]["cy"]
    for d in pads_by_y[1:]:
        if d["cy"] - row_ref_y <= row_tol:
            cur_row.append(d)
        else:
            rows.append(cur_row)
            cur_row = [d]
        row_ref_y = d["cy"]  # 用上一个点的y做滚动基准,适应整体倾斜

    rows.append(cur_row)

    # 4. 逐行往返(蛇形): 行内按x排序,奇数行反向
    ordered_pads = []
    for ri, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda d: d["cx"])
        if ri % 2 == 1:
            row_sorted = row_sorted[::-1]
        ordered_pads.extend(row_sorted)

    # 5. 按排好的焊盘顺序拼接各组的点; 每进入一个焊盘, 若其末端离上一结束点更近则整体翻转,
    #    使焊盘间衔接走最近端, 消除焊盘内填充序列方向不一致导致的斜向回跳。
    ordered = []
    last = None
    for d in ordered_pads:
        g = groups[d["pid"]]
        if last is not None and len(g) > 1:
            d0 = (g[0]["x"] - last["x"]) ** 2 + (g[0]["y"] - last["y"]) ** 2
            d1 = (g[-1]["x"] - last["x"]) ** 2 + (g[-1]["y"] - last["y"]) ** 2
            if d1 < d0:
                g = g[::-1]
        ordered.extend(g)
        last = ordered[-1]

    return ordered


def path_to_gcode(path, z_travel=Z_TRAVEL, z_dispense=Z_DISPENSE, speed=MOVE_SPEED):
    """
    将路径转为G-code
    """
    lines = []
    lines.append("; === 点锡路径 G-code ===")
    lines.append(f"; 总点数: {len(path)}")
    lines.append("G90 ; 绝对坐标")
    lines.append("G21 ; 毫米单位")
    lines.append(f"G0 Z{z_travel:.2f} ; 抬起到安全高度")
    
    for i, pt in enumerate(path):
        px, py = pixel_to_physical([[pt["x"], pt["y"]]])[0]
        lines.append(f"; Point {i}: class={pt['class']}, type={pt['type']}")
        lines.append(f"G0 X{px:.3f} Y{py:.3f} Z{z_travel:.2f} F{speed*60}")
        lines.append(f"G1 Z{z_dispense:.2f} F{speed*30}")
        lines.append(f"G4 P{pt['dwell']} ; 点锡停留")
        lines.append(f"G0 Z{z_travel:.2f} ; 抬起")
    
    lines.append("G0 X0 Y0 Z{:.2f} ; 回原点".format(z_travel))
    lines.append("; === END ===")
    return "\n".join(lines)


def generate_path(bboxes, scores, class_ids, output_json=None, output_gcode=None):
    """
    主函数：从检测结果生成优化路径
    
    Returns:
        dict: {"points": [...], "total": int, "gcode": str}
    """
    # 1. 生成点锡坐标
    points = generate_dispense_points(bboxes, scores, class_ids)
    print(f"[PathGen] 生成 {len(points)} 个点锡点 (来自 {len(bboxes)} 个检测目标)")
    
    # 2. 路径优化
    optimized = optimize_path(points)
    
    # 3. 计算总路程
    total_dist = 0
    for i in range(1, len(optimized)):
        dx = optimized[i]["x"] - optimized[i-1]["x"]
        dy = optimized[i]["y"] - optimized[i-1]["y"]
        total_dist += np.hypot(dx, dy)
    print(f"[PathGen] 路径优化完成, 总行程: {total_dist:.1f} px")
    
    # 4. G-code
    gcode = path_to_gcode(optimized)
    
    result = {
        "points": optimized,
        "total_points": len(optimized),
        "total_distance_px": round(total_dist, 1),
        "gcode": gcode
    }
    
    # 保存文件
    if output_json:
        with open(output_json, 'w') as f:
            json.dump({"points": optimized, "total_points": len(optimized), 
                      "total_distance_px": round(total_dist, 1)}, f, indent=2)
        print(f"[PathGen] JSON路径已保存: {output_json}")
    
    if output_gcode:
        with open(output_gcode, 'w') as f:
            f.write(gcode)
        print(f"[PathGen] G-code已保存: {output_gcode}")
    
    return result


def visualize_path(img, path, output_path=None):
    """在图像上可视化点锡路径"""
    vis = img.copy()
    
    for i, pt in enumerate(path):
        x, y = int(pt["x"]), int(pt["y"])
        color = (0, 0, 255) if pt["type"] == "single" else (255, 0, 0)
        cv2.circle(vis, (x, y), 3, color, -1)
        
        # 画路径连线
        if i > 0:
            px, py = int(path[i-1]["x"]), int(path[i-1]["y"])
            cv2.line(vis, (px, py), (x, y), (0, 255, 255), 1)
    
    # 标注起点和终点
    if len(path) > 0:
        sx, sy = int(path[0]["x"]), int(path[0]["y"])
        cv2.circle(vis, (sx, sy), 8, (0, 255, 0), 2)
        cv2.putText(vis, "START", (sx+5, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        ex, ey = int(path[-1]["x"]), int(path[-1]["y"])
        cv2.circle(vis, (ex, ey), 8, (0, 0, 255), 2)
        cv2.putText(vis, "END", (ex+5, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"[PathGen] 路径可视化已保存: {output_path}")
    
    return vis


# === 独立测试入口 ===
if __name__ == "__main__":
    # 配合 infer.py 使用示例
    from infer import main as infer_main
    import sys
    
    # 也可以直接传入检测结果测试
    print("=== 点锡路径生成器测试 ===")
    # 模拟几个检测框
    test_bboxes = np.array([
        [100, 100, 130, 130],  # 小焊盘
        [200, 150, 280, 230],  # 大焊盘
        [350, 100, 380, 125],  # 小焊盘
        [400, 300, 420, 320],  # 小焊盘
    ])
    test_scores = np.array([0.9, 0.85, 0.8, 0.75])
    test_class_ids = np.array([0, 1, 0, 2])
    
    result = generate_path(
        test_bboxes, test_scores, test_class_ids,
        output_json="/home/elf/yolo/path_output.json",
        output_gcode="/home/elf/yolo/path_output.gcode"
    )
    print(f"\n总点数: {result['total_points']}")
    print(f"总行程: {result['total_distance_px']} px")
    print(f"\nG-code预览(前10行):")
    for line in result['gcode'].split("\n")[:10]:
        print(f"  {line}")
