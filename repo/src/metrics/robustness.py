"""
robustness.py
Metrica de robustez:
- VLM: sensibilidad al prompt (varianza de F1 entre variantes de redaccion).
- YOLO (cerrado): rendimiento en una clase o escena que no entreno (zero-shot fail).
"""

import numpy as np


def compute_prompt_sensitivity(f1_per_prompt: dict) -> dict:
    """
    f1_per_prompt: {prompt_text: f1_score} para cada variante de redaccion.
    Devuelve media, desviacion estandar y rango (max - min) del F1 entre prompts.
    Reportar varianza, no un unico numero (ver restricciones del enunciado).
    """
    scores = np.array(list(f1_per_prompt.values()))
    return {
        "mean_f1": float(np.mean(scores)),
        "std_f1": float(np.std(scores)),
        "min_f1": float(np.min(scores)),
        "max_f1": float(np.max(scores)),
        "range_f1": float(np.max(scores) - np.min(scores)),
        "per_prompt": f1_per_prompt,
    }


def compute_unseen_class_performance(detections: list, unseen_class: str) -> dict:
    """
    Mide cuantos frames el detector CERRADO (YOLOv8n) logra detectar una clase
    o escena que no fue parte de su entrenamiento. Se espera un desempeno bajo
    (o nulo), evidenciando la limitacion del paradigma cerrado.

    detections: lista de dicts con "classes" (clases detectadas por frame).
    """
    frames_with_detection = sum(
        1 for d in detections if unseen_class in d.get("classes", [])
    )
    total_frames = len(detections)
    detection_rate = frames_with_detection / total_frames if total_frames > 0 else 0.0

    return {
        "unseen_class": unseen_class,
        "frames_with_detection": frames_with_detection,
        "total_frames": total_frames,
        "detection_rate": detection_rate,
    }
