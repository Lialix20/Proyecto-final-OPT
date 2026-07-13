"""
event_metrics.py
Metrica nivel-evento: F1 sobre la pregunta binaria por frame
"¿hay alguien sin casco?" (GT = el frame contiene una caja NO-Hardhat).
Esta es la comparacion justa entre los tres paradigmas (el hallazgo central
del proyecto: la negacion).
"""

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def frame_has_violation_gt(gt_classes: list, violation_class: str = "NO-Hardhat") -> bool:
    """True si el ground truth del frame contiene la clase de violacion."""
    return violation_class in gt_classes


def frame_has_violation_pred_boxes(pred_classes: list, violation_class: str = "NO-Hardhat") -> bool:
    """
    Para modelos que devuelven cajas con clase (YOLOv8n, YOLO-World, Florence-2):
    True si alguna deteccion corresponde a la clase de violacion.
    """
    return violation_class in pred_classes


def compute_event_level_f1(y_true: list, y_pred: list) -> dict:
    """
    y_true, y_pred: listas de booleanos (uno por frame), en el MISMO orden.
    Devuelve F1, precision, recall y matriz de confusion.
    """
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }
