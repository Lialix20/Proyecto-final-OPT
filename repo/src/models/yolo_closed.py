"""
yolo_closed.py
Paradigma 1: Detector CERRADO (clases fijas, entrenadas).
Fine-tuning de YOLOv8n sobre CSS-Data e inferencia sobre el split test.
Licencia: AGPL-3.0 (Ultralytics).
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    CHECKPOINTS_DIR, YOLO_EPOCHS, YOLO_IMG_SIZE, YOLO_BATCH,
    YOLO_MODEL_NAME, SEED,
)


def train_yolo_closed(data_yaml_path: str, epochs: int = YOLO_EPOCHS,
                       imgsz: int = YOLO_IMG_SIZE, batch: int = YOLO_BATCH):
    """
    Afina YOLOv8n sobre CSS-Data (data.yaml generado por Roboflow).
    Devuelve el modelo entrenado y la ruta al mejor checkpoint (best.pt).
    """
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL_NAME)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=SEED,
        project=CHECKPOINTS_DIR,
        name="yolov8n_css_data",
        exist_ok=True,
        patience=15,
        verbose=True,
    )

    best_ckpt = os.path.join(CHECKPOINTS_DIR, "yolov8n_css_data", "weights", "best.pt")
    print(f"[yolo_closed] Entrenamiento terminado. Mejor checkpoint: {best_ckpt}")
    return model, best_ckpt


def evaluate_yolo_closed(checkpoint_path: str, data_yaml_path: str):
    """Evalua el modelo entrenado sobre el split test (mAP@0.5 nativo de Ultralytics)."""
    from ultralytics import YOLO

    model = YOLO(checkpoint_path)
    metrics = model.val(data=data_yaml_path, split="test", imgsz=YOLO_IMG_SIZE)
    return metrics


def predict_yolo_closed(checkpoint_path: str, image_paths: list, conf: float = 0.25):
    """
    Corre inferencia frame por frame y mide latencia individual.
    Devuelve una lista de dicts: {frame_id, boxes, classes, scores, latency_s}.
    """
    from ultralytics import YOLO

    model = YOLO(checkpoint_path)
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

    print(f"[yolo_closed] Inferencia completada sobre {len(image_paths)} frames")
    return results_out
