#!/usr/bin/env python3
"""
Gerber 锡膏层(Paste)焊盘解析器 — 用于点锡机自动取点 + 板子朝向对位模板
====================================================================
功能:
  - 输入 Gerber zip 或已解压目录
  - 解析顶层锡膏层(.GTP)的全部焊盘开窗:
        * D03 flash (aperture R矩形/O椭圆/C圆) -> 中心+尺寸
        * G36/G37 多边形填充           -> 包围盒中心+尺寸
  - 按规则过滤: 长或宽 < MIN_SIDE_MM(默认0.5) 的焊盘不贴
  - 输出焊盘清单(机床无关的设计坐标, mm) + json + 可视化png

坐标格式: EasyEDA Pro FSLAX45Y45 (整数/1e5 = mm)。若其他工具导出可调 COORD_DIV。
注意: 这里给出的是 PCB 设计坐标(gerber原点), 上点锡机时需再经过
      "焊盘点集 <-> 相机识别点集" 配准, 解出板子在机床上的平移+旋转。
"""
import os
import re
import json
import zipfile
import tempfile

COORD_DIV = 1e5          # FSLAX45Y45
MIN_SIDE_MM = 0.5        # 长或宽小于此值不贴


def _find_layer(root, exts):
    """在目录中找指定后缀的层文件(不区分大小写)。"""
    for dp, _, fns in os.walk(root):
        for f in fns:
            if os.path.splitext(f)[1].lower() in exts:
                return os.path.join(dp, f)
    return None


def _parse_apertures(txt):
    """解析 %ADDnn{C|R|O|P},dim[XdimX..]*% -> {dcode: (shape, [dims])}。"""
    aps = {}
    for ad in re.finditer(r'%ADD(\d+)([CROP]),([0-9.X]+)\*%', txt):
        dims = [float(x) for x in ad.group(3).split('X')]
        aps[int(ad.group(1))] = (ad.group(2), dims)
    return aps


def parse_paste_pads(gtp_path):
    """解析锡膏层全部焊盘。返回 [(x_mm, y_mm, w_mm, h_mm, shape), ...]。"""
    txt = open(gtp_path, encoding='utf-8', errors='ignore').read()
    aps = _parse_apertures(txt)
    pads = []

    # 1) D03 flash: 行内含 ...X..Y..D03* , 用当前选中的 aperture 尺寸
    cur = None
    for line in txt.splitlines():
        s = line.strip()
        m = re.fullmatch(r'(?:G54)?D(\d+)\*', s)
        if m and int(m.group(1)) >= 10:
            cur = int(m.group(1))
            continue
        f = re.search(r'X(-?\d+)Y(-?\d+)D03\*', s)
        if f:
            x = int(f.group(1)) / COORD_DIV
            y = int(f.group(2)) / COORD_DIV
            shape, dims = aps.get(cur, ('?', [0.0, 0.0]))
            w = dims[0]
            h = dims[1] if len(dims) > 1 else dims[0]
            pads.append((x, y, w, h, shape))

    # 2) G36/G37 多边形填充: 取顶点包围盒
    for blk in re.findall(r'G36\*(.*?)G37\*', txt, re.S):
        pts = re.findall(r'X(-?\d+)Y(-?\d+)D0[12]\*', blk)
        if len(pts) >= 3:
            xs = [int(a) / COORD_DIV for a, b in pts]
            ys = [int(b) / COORD_DIV for a, b in pts]
            pads.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                         max(xs) - min(xs), max(ys) - min(ys), 'POLY'))
    return pads


def filter_pads(pads, min_side=MIN_SIDE_MM):
    """长或宽 < min_side 的焊盘剔除(不贴)。返回 (keep, drop)。"""
    keep = [p for p in pads if min(p[2], p[3]) >= min_side]
    drop = [p for p in pads if min(p[2], p[3]) < min_side]
    return keep, drop


def load_gerber(path):
    """输入 zip 或目录, 返回解压后目录(zip解压到临时目录)。"""
    if os.path.isdir(path):
        return path, None
    tmp = tempfile.mkdtemp(prefix='gerber_')
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    return tmp, tmp


def parse_board_outline(root):
    """解析板框层(.gko/.gm1)的坐标包围盒, 返回(minx,miny,maxx,maxy)mm或None。"""
    p = _find_layer(root, {'.gko', '.gm1'})
    if not p:
        return None
    txt = open(p, encoding='utf-8', errors='ignore').read()
    xs, ys = [], []
    for m in re.finditer(r'X(-?\d+)Y(-?\d+)D0[12]\*', txt):
        xs.append(int(m.group(1)) / COORD_DIV)
        ys.append(int(m.group(2)) / COORD_DIV)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def extract_paste_targets(gerber_path, min_side=MIN_SIDE_MM, layer_ext=('.gtp',), mirror_x=False):
    """主入口: 输入gerber(zip/目录) -> dict(含保留焊盘点集+统计+板框)。
    mirror_x=True: 底面用, 焊盘绕板框中心X镜像(翻板后视图与gerber底层镜像关系)。"""
    root, _ = load_gerber(gerber_path)
    gtp = _find_layer(root, set(layer_ext))
    if not gtp:
        raise FileNotFoundError(f"未找到锡膏层{layer_ext}: {root}")
    pads = parse_paste_pads(gtp)
    keep, drop = filter_pads(pads, min_side)
    outline = parse_board_outline(root)
    # 底面镜像: 绕板框中心X翻转(无板框则用焊盘包络中心)
    if mirror_x and keep:
        if outline:
            cx = (outline[0] + outline[2]) / 2.0
        else:
            xs = [p[0] for p in keep]
            cx = (min(xs) + max(xs)) / 2.0
        keep = [(2 * cx - x, y, w, h, sh) for (x, y, w, h, sh) in keep]
        if outline:
            outline = (2 * cx - outline[2], outline[1], 2 * cx - outline[0], outline[3])
    return {
        'layer_file': os.path.basename(gtp),
        'min_side_mm': min_side,
        'total': len(pads),
        'keep': len(keep),
        'drop': len(drop),
        'mirrored': bool(mirror_x),
        'outline': outline,
        'pads': [{'x': round(x, 4), 'y': round(y, 4),
                  'w': round(w, 4), 'h': round(h, 4), 'shape': sh}
                 for (x, y, w, h, sh) in keep],
    }


def save_json(result, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def visualize(gerber_path, out_png, min_side=MIN_SIDE_MM, board_mm=None):
    """画保留(绿)与剔除(灰)焊盘分布, 便于核对。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root, _ = load_gerber(gerber_path)
    gtp = _find_layer(root, {'.gtp'})
    pads = parse_paste_pads(gtp)
    keep, drop = filter_pads(pads, min_side)

    fig, ax = plt.subplots(figsize=(7, 7))
    if board_mm:
        ax.add_patch(plt.Rectangle((0, 0), board_mm[0], board_mm[1],
                                   fill=False, edgecolor='green', lw=1.5))
    for x, y, w, h, _ in drop:
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), max(w, 0.3), max(h, 0.3),
                                   facecolor='#ccc', edgecolor='#999', lw=0.3))
    for x, y, w, h, _ in keep:
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), max(w, 0.4), max(h, 0.4),
                                   facecolor='#34c759', edgecolor='#1a7', lw=0.4))
    allx = [p[0] for p in pads] or [0]
    ally = [p[1] for p in pads] or [0]
    ax.scatter([], [], c='#34c759', marker='s', label=f'贴 keep {len(keep)}')
    ax.scatter([], [], c='#ccc', marker='s', label=f'不贴 drop {len(drop)} (<{min_side}mm)')
    ax.set_xlim(min(allx) - 5, max(allx) + 5)
    ax.set_ylim(min(ally) - 5, max(ally) + 5)
    ax.set_aspect('equal')
    ax.set_title(f'Paste pads: keep {len(keep)} / total {len(pads)}')
    ax.set_xlabel('X mm'); ax.set_ylabel('Y mm'); ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close()
    return out_png


if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser(r'~\Desktop\Gerber_PCB1_1_2026-06-28.zip')
    res = extract_paste_targets(src)
    print(f"锡膏层: {res['layer_file']}")
    print(f"焊盘总数 {res['total']}, 贴 {res['keep']}, 不贴 {res['drop']} (<{res['min_side_mm']}mm)")
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paste_targets.json')
    save_json(res, out_json)
    print('焊盘清单已存:', out_json)
