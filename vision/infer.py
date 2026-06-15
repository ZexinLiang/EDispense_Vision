
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import time

MODEL_PATH = '/home/elf/yolo/material-640-640-v5n.rknn'
IMG_PATH = '/home/elf/yolo/1.jpg'
OUTPUT_PATH = '/home/elf/yolo/result.jpg'

CONF_THRESH = 0.25
NMS_THRESH = 0.45
INPUT_SIZE = 640

def letterbox(img, new_shape=(640, 640)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_output(outputs, img_shape, r, pad):
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output[{i}]: shape={out.shape}, dtype={out.dtype}, min={out.min():.3f}, max={out.max():.3f}")

    # Single output: [1, N, 5+C] or [N, 5+C]
    if len(outputs) == 1:
        out = outputs[0]
        if len(out.shape) == 3:
            out = out[0]
        boxes = out[:, :4]
        obj_conf = out[:, 4]
        class_probs = out[:, 5:]

        if obj_conf.max() > 1.0 or obj_conf.min() < 0.0:
            obj_conf = sigmoid(obj_conf)
            class_probs = sigmoid(class_probs)

        class_ids = np.argmax(class_probs, axis=1)
        class_scores = np.max(class_probs, axis=1)
        scores = obj_conf * class_scores

        mask = scores > CONF_THRESH
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            print("No detections above threshold")
            return [], [], []

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        dw, dh = pad
        x1 = (x1 - dw) / r
        y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r
        y2 = (y2 - dh) / r

        bboxes = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(), scores.tolist(), CONF_THRESH, NMS_THRESH
        )
        if len(indices) > 0:
            indices = indices.flatten()
            return bboxes[indices], scores[indices], class_ids[indices]
        return [], [], []

    # Multi-output from RKNN YOLOv5: 3 heads
    # Output shape: (1, 3*nc_per_anchor, H, W) where nc_per_anchor = 5 + num_classes
    # For this model: (1, 24, 80, 80), (1, 24, 40, 40), (1, 24, 20, 20)
    # 24 = 3 anchors * (4 bbox + 1 obj + 3 classes) = 3 * 8
    print("Multi-head output, decoding...")
    
    # YOLOv5n anchors (standard)
    anchors = [
        [[10,13],[16,30],[33,23]],    # P3/8
        [[30,61],[62,45],[59,119]],    # P4/16
        [[116,90],[156,198],[373,326]] # P5/32
    ]
    strides = [8, 16, 32]
    
    all_boxes = []
    all_scores = []
    all_class_ids = []
    
    for idx, out in enumerate(outputs):
        # Remove batch dim: (1, 24, H, W) -> (24, H, W)
        if len(out.shape) == 4:
            out = out[0]
        
        c, h, w = out.shape
        na = 3  # number of anchors
        nc = c // na  # channels per anchor = 5 + num_classes
        num_classes = nc - 5
        
        # Reshape: (24, H, W) -> (3, 8, H, W) -> (3, H, W, 8)
        out = out.reshape(na, nc, h, w).transpose(0, 2, 3, 1)
        # Now: (3, H, W, 8) where last dim = [x, y, w, h, obj, cls0, cls1, cls2]
        
        stride = strides[idx]
        anchor = np.array(anchors[idx])
        
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        for a in range(na):
            data = out[a]  # (H, W, 8)
            
            # Data is already sigmoid-ed (values in 0-1 range from RKNN output)
            bx = (data[..., 0] * 2 - 0.5 + grid_x) * stride
            by = (data[..., 1] * 2 - 0.5 + grid_y) * stride
            bw = (data[..., 2] * 2) ** 2 * anchor[a][0]
            bh = (data[..., 3] * 2) ** 2 * anchor[a][1]
            obj = data[..., 4]
            cls = data[..., 5:]
            
            cls_id = np.argmax(cls, axis=-1)
            cls_score = np.max(cls, axis=-1)
            score = obj * cls_score
            
            mask = score > CONF_THRESH
            if not np.any(mask):
                continue
            
            bx_f = bx[mask]
            by_f = by[mask]
            bw_f = bw[mask]
            bh_f = bh[mask]
            
            # Convert to original image coords
            dw, dh = pad
            x1 = (bx_f - bw_f/2 - dw) / r
            y1 = (by_f - bh_f/2 - dh) / r
            x2 = (bx_f + bw_f/2 - dw) / r
            y2 = (by_f + bh_f/2 - dh) / r
            
            all_boxes.extend(np.stack([x1,y1,x2,y2], axis=1).tolist())
            all_scores.extend(score[mask].tolist())
            all_class_ids.extend(cls_id[mask].tolist())
    
    if len(all_boxes) == 0:
        print("No detections from multi-head")
        return [], [], []
    
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_class_ids = np.array(all_class_ids)
    
    print(f"  Pre-NMS detections: {len(all_boxes)}")
    indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), CONF_THRESH, NMS_THRESH)
    if len(indices) > 0:
        indices = indices.flatten()
        return all_boxes[indices], all_scores[indices], all_class_ids[indices]
    return [], [], []

def main():
    print("=== YOLOv5 RKNN Inference ===")

    rknn = RKNNLite()

    print(f"Loading model: {MODEL_PATH}")
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"Load model failed! ret={ret}")
        return

    print("Init runtime...")
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"Init runtime failed! ret={ret}")
        return

    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Failed to read image: {IMG_PATH}")
        return
    print(f"Input image: {img.shape}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, r, pad = letterbox(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    print(f"Preprocessed: {img_resized.shape}, ratio={r:.3f}, pad={pad}")

    # Add batch dimension: (640,640,3) -> (1,640,640,3)
    img_input = np.expand_dims(img_resized, axis=0)
    print(f"Input tensor shape: {img_input.shape}")

    print("Running inference...")
    t0 = time.time()
    outputs = rknn.inference(inputs=[img_input])
    t1 = time.time()
    print(f"Inference time: {(t1-t0)*1000:.1f} ms")

    bboxes, scores, class_ids = process_output(outputs, img.shape[:2], r, pad)

    if len(bboxes) > 0:
        print(f"\nDetections: {len(bboxes)}")
        for i, (box, score, cls) in enumerate(zip(bboxes, scores, class_ids)):
            x1, y1, x2, y2 = [int(v) for v in box]
            print(f"  [{i}] class={int(cls)}, conf={score:.3f}, box=[{x1},{y1},{x2},{y2}]")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{int(cls)}:{score:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(OUTPUT_PATH, img)
        print(f"Result saved: {OUTPUT_PATH}")
    else:
        print("No detections")
        cv2.imwrite(OUTPUT_PATH, img)

    rknn.release()
    print("Done!")

if __name__ == '__main__':
    main()
