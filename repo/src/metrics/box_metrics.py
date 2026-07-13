"""
box_metrics.py
Metricas nivel-caja: IoU y mAP@0.5, sobre las clases compartibles
(Person, Hardhat) entre los paradigmas que devuelven cajas
(YOLOv8n, YOLO-World, Florence-2).
"""

import numpy as np


def compute_iou(box_a, box_b):
    """IoU entre dos cajas [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def match_detections_to_gt(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
    """
    Empareja predicciones con ground truth por clase e IoU >= threshold.
    Devuelve (tp, fp, fn) a nivel de conteo para un frame.
    """
    matched_gt = set()
    tp = 0

    for p_box, p_cls in zip(pred_boxes, pred_classes):
        best_iou, best_idx = 0.0, -1
        for i, (g_box, g_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if i in matched_gt or g_cls != p_cls:
                continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_iou >= iou_threshold and best_idx != -1:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compute_precision_recall_curve(all_preds, all_gts, class_name, iou_threshold=0.5):
    """
    Curva precision-recall simplificada para una clase, ordenando por score
    descendente. all_preds: lista de dicts {boxes, classes, scores}.
    all_gts: lista de dicts {boxes, classes} (mismo orden de frames).
    """
    detections = []  # (score, is_tp)
    total_gt = 0

    for pred, gt in zip(all_preds, all_gts):
        gt_boxes_cls = [b for b, c in zip(gt["boxes"], gt["classes"]) if c == class_name]
        total_gt += len(gt_boxes_cls)

        pred_items = [
            (score, box) for box, cls, score in
            zip(pred["boxes"], pred["classes"], pred.get("scores", [1.0] * len(pred["boxes"])))
            if cls == class_name
        ]
        pred_items.sort(key=lambda x: x[0], reverse=True)

        matched = [False] * len(gt_boxes_cls)
        for score, p_box in pred_items:
            best_iou, best_idx = 0.0, -1
            for i, g_box in enumerate(gt_boxes_cls):
                if matched[i]:
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            is_tp = best_iou >= iou_threshold and best_idx != -1
            if is_tp:
                matched[best_idx] = True
            detections.append((score, is_tp))

    detections.sort(key=lambda x: x[0], reverse=True)
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []

    for score, is_tp in detections:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt if total_gt > 0 else 0.0)

    return np.array(precisions), np.array(recalls)


def compute_ap(precisions, recalls):
    """Average Precision por integracion tipo VOC (11 puntos simplificado a interpolacion continua)."""
    if len(precisions) == 0:
        return 0.0
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return float(ap)


def compute_map50(all_preds, all_gts, class_names, iou_threshold=0.5):
    """mAP@0.5 promediado sobre las clases dadas."""
    aps = {}
    for cls in class_names:
        precisions, recalls = compute_precision_recall_curve(all_preds, all_gts, cls, iou_threshold)
        aps[cls] = compute_ap(precisions, recalls)

    mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
    return mean_ap, aps
