#!/usr/bin/env python3
"""
RKNN YOLOv5 推理模块 (1088输入, 移植自转模型验证代码 test-1088-1088.py)
=====================================================================
预处理: 输入正方形ROI(1080x1080) → 直接resize到1088x1088 (BGR→RGB)
后处理: rknn_model_zoo官方yolov5后处理, 按类别分组NMS(不同类不互相抑制)
类别: 0=hole, 1=pad, 2=qfn
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import time

MODEL_PATH = '/home/elf/solder_system/models/pad.rknn'
CONF_THRESH = 0.25      # OBJ_THRESH
NMS_THRESH = 0.45
INPUT_SIZE = 1088
CLASSES = ("hole", "pad", "qfn")
ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
           [59, 119], [116, 90], [156, 198], [373, 326]]


def xywh2xyxy(x):
    """[cx,cy,w,h] -> [x1,y1,x2,y2]"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _process(input, mask, anchors):
    """解码单个检测头。input: (grid_h, grid_w, 3, 5+nc)"""
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = input[..., 5:]

    box_xy = input[..., :2] * 2 - 0.5
    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(INPUT_SIZE / grid_h)

    box_wh = pow(input[..., 2:4] * 2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)
    return box, box_confidence, box_class_probs


def _filter_boxes(boxes, box_confidences, box_class_probs):
    """两道阈值过滤: obj_conf + class_score"""
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= CONF_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= CONF_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]
    return boxes, classes, scores


def _nms_boxes(boxes, scores):
    """单类别NMS"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


def yolov5_post_process(input_data):
    """完整后处理: 解码3头 + 按类别分组NMS。返回(boxes,classes,scores)或(None,None,None)"""
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    boxes, classes, scores = [], [], []
    for inp, mask in zip(input_data, masks):
        b, c, s = _process(inp, mask, ANCHORS)
        b, c, s = _filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        cc = classes[inds]
        s = scores[inds]
        keep = _nms_boxes(b, s)
        nboxes.append(b[keep])
        nclasses.append(cc[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores


def preprocess(roi_bgr):
    """正方形ROI(BGR) → 1088x1088 RGB, 直接resize(轻微拉伸)。
    返回 (img_input, scale) ; scale = ROI边长/1088, 用于坐标反算回ROI"""
    side = roi_bgr.shape[0]  # 正方形, H==W
    img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    scale = side / INPUT_SIZE
    return img, scale


def reshape_outputs(outputs):
    """RKNN三头输出 (1,24,H,W) → list of (H,W,3,8) 供后处理"""
    input_data = []
    for out in outputs:
        # (1,24,H,W) -> (3,8,H,W)
        d = out.reshape([3, -1] + list(out.shape[-2:]))
        # -> (H,W,3,8)
        input_data.append(np.transpose(d, (2, 3, 0, 1)))
    return input_data


def infer(rknn, roi_bgr, conf_thresh=None):
    """完整推理: ROI(正方形BGR) → 检测结果(映射回ROI像素坐标)
    返回 bboxes[N,4](x1y1x2y2), scores[N], class_ids[N]"""
    global CONF_THRESH
    if conf_thresh is not None:
        CONF_THRESH = float(conf_thresh)
    img, scale = preprocess(roi_bgr)
    img_input = np.expand_dims(img, 0)
    outputs = rknn.inference(inputs=[img_input], data_format=['nhwc'])
    input_data = reshape_outputs(outputs)
    boxes, classes, scores = yolov5_post_process(input_data)
    if boxes is None:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    # 坐标从1088空间反算回ROI空间
    boxes = boxes * scale
    return boxes, scores, classes.astype(int)


def main():
    """命令行测试入口：对指定图片跑推理并打印检测结果"""
    import sys
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime()
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else '/home/elf/solder_system/data/1.jpg')
    h, w = img.shape[:2]
    side = min(h, w)
    roi = img[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
    t0 = time.time()
    boxes, scores, cls = infer(rknn, roi)
    print(f"infer {(time.time()-t0)*1000:.0f}ms, {len(boxes)} dets")
    for b, s, c in zip(boxes, scores, cls):
        print(f"  {CLASSES[c]} {s:.2f} {[int(v) for v in b]}")
    rknn.release()


if __name__ == '__main__':
    main()
