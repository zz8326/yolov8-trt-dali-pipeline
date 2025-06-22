import numpy as np
from utils import load_class_names



def _postprocess_end2end(data, batch_ratios, class_names, batch_size=1, score_thresh=0.25):
    results = []
    batch_num, batch_boxes, batch_scores, batch_cls_inds = data
    batch_boxes = batch_boxes.reshape(batch_size, -1, 4) # (1, 100, 4)
    batch_scores = batch_scores.reshape(batch_size, -1) # (1, 100)
    batch_cls_inds = batch_cls_inds.reshape(batch_size, -1) # (1, 100)
 
    for b, ratio in enumerate(batch_ratios):
        n = batch_num[b] 
        boxes = batch_boxes[b, :n] / ratio # 還原框
        scores = batch_scores[b, :n] # 定義分數
        classes = batch_cls_inds[b, :n] # class id

        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            class_id = int(cls)
            if score < score_thresh:
                continue
            x0, y0, x1, y1 = map(int, box)
            detections.append({
                "bbox": [x0, y0, x1, y1],
                "score": float(score),
                "class_id": class_names[class_id]
            })
        results.append(detections)
    return results


def _postprocess_yolov8_raw(data, batch_ratios, class_names, batch_size=1, num_classes=80, score_thresh=0.25):
    """
        output = (1, 84, 8400) 84 = 4(xywh) + 80(classses) 
    """
    results = []
    data_array = data[0].reshape(batch_size, 4 + num_classes, -1)
    data_array = np.transpose(data_array, (0, 2, 1))#(N, bbox_count, xywh)


    for idx, ratio in enumerate(batch_ratios):
        preds = data_array[idx] # (8400, 84)
        boxes = preds[:, :4]
        scores = preds[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        valid = confidences > score_thresh
        boxes = boxes[valid]
        confidences = confidences[valid]
        class_ids = class_ids[valid]

        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, confidences, class_ids)

        image_detections = []
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            for final_box, score, cls in zip(final_boxes, final_scores, final_cls_inds):
                x0, y0, x1, y1 = map(int, final_box)
                cls = int(cls)
                image_detections.append({
                    "bbox": [x0, y0, x1, y1],
                    "score": float(score),
                    "class_id": class_names[cls]
                })
        results.append(image_detections)
    return results

def postprocess_batch(data, batch_ratios, batch_size=1,num_classes=80, score_thresh=0.25,end2end=False,):
    class_names = load_class_names("./config/meta_det.yaml")
    if end2end:
       dets = _postprocess_end2end(data, batch_ratios, class_names=class_names, batch_size=batch_size, score_thresh=score_thresh)
    else:
       dets =  _postprocess_yolov8_raw(data, batch_ratios, class_names = class_names, batch_size=batch_size, num_classes=num_classes, score_thresh=score_thresh)   
    
    return dets

def nms(boxes, scores, iou_thresh):
    """
    boxes: numpy array of [N, 4] - (x1, y1, x2, y2)
    scores: numpy array of [N]
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 框的面積
    order = scores.argsort()[::-1]  # 根據 score 大到小排序 index

    keep = []

    while order.size > 0:
        i = order[0]  # 分數最高的 index
        keep.append(i)

        # 和分數第2名以後的框計算 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union

        # 刪掉那些跟目前框 i 重疊太多的
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]  # +1 是因為第 0 是自己

    return keep

def multiclass_nms(boxes, scores, class_ids, iou_thresh=0.45, score_thresh=0.5):
    final_dets = []
    unique_classes = np.unique(class_ids)

    for cls in unique_classes:
        idxs = np.where(class_ids == cls)[0]
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]

        valid = cls_scores > score_thresh
        if valid.sum() == 0:
            continue

        keep = nms(cls_boxes[valid], cls_scores[valid], iou_thresh)

        if len(keep) > 0:
            kept_boxes = cls_boxes[valid][keep]
            kept_scores = cls_scores[valid][keep]
            kept_classes = np.full((len(keep), 1), cls)

            det = np.concatenate([kept_boxes, kept_scores[:, None], kept_classes], axis=1)
            final_dets.append(det)

    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, axis=0)




if __name__ == '__main__':
    from infer.trt_v8 import TensorRTv8
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    model = TensorRTv8("./model/yolov8n_640_batch1_end2end_fp16.engine", ctx=ctx)
    ctx.pop()

        
