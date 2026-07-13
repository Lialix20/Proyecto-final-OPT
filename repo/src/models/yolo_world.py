"""
yolo_world.py
Paradigma 2: Vocabulario ABIERTO. YOLO-World recibe las clases como texto
en inferencia (set_classes), sin necesidad de reentrenar (zero-shot).
Licencia: GPL-3.0 (Ultralytics).
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import YOLO_WORLD_PROMPTS


def load_yolo_world(prompts: list = None):
    """Carga YOLO-World y fija el vocabulario abierto (clases de texto)."""
    from ultralytics import YOLO

    prompts = prompts or YOLO_WORLD_PROMPTS
    model = YOLO("yolov8s-world.pt")
    model.set_classes(prompts)
    print(f"[yolo_world] Vocabulario fijado: {prompts}")
    return model


def predict_yolo_world(model, image_paths: list, conf: float = 0.1):
    """
    Corre inferencia zero-shot frame por frame y mide latencia.
    Devuelve lista de dicts: {frame_id, boxes, classes, scores, latency_s}.
    """
    results_out = []

    for img_path in image_paths:
        frame_id = os.path.splitext(os.path.basename(img_path))[0]

        t0 = time.perf_counter()
        result = model.predict(img_path, conf=conf, verbose=False)[0]
        latency = time.perf_counter() - t0

        boxes = result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) else []
        classes = [result.names[int(c)] for c in result.boxes.cls.cpu().numpy()] if len(result.boxes) else []
        scores = result.boxes.conf.cpu().numpy().tolist() if len(result.boxes) else []

        results_out.append({
            "frame_id": frame_id,
            "boxes": boxes,
            "classes": classes,
            "scores": scores,
            "latency_s": latency,
        })

    print(f"[yolo_world] Inferencia completada sobre {len(image_paths)} frames")
    return results_out
