"""
florence2.py
Paradigma 3 (VLM de grounding): Florence-2 (Microsoft, MIT).
Tarea <OPEN_VOCABULARY_DETECTION>: se entrega un texto y devuelve cajas.
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import FLORENCE2_MODEL_ID, DEVICE


def load_florence2():
    """Carga Florence-2 y su procesador desde HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE2_MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

    print(f"[florence2] Modelo cargado: {FLORENCE2_MODEL_ID}")
    return model, processor


def predict_florence2(model, processor, image_paths: list, text_prompt: str = "hard hat. person. head."):
    """
    Corre <OPEN_VOCABULARY_DETECTION> frame por frame.
    text_prompt: clases separadas por punto, en lenguaje natural.
    Devuelve lista de dicts: {frame_id, boxes, classes, latency_s}.
    """
    import torch
    from PIL import Image

    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    results_out = []

    for img_path in image_paths:
        frame_id = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert("RGB")

        inputs = processor(text=task_prompt + text_prompt, images=image, return_tensors="pt").to(
            DEVICE, torch.float16
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
        latency = time.perf_counter() - t0

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        detections = parsed.get(task_prompt, {})
        boxes = detections.get("bboxes", [])
        classes = detections.get("bboxes_labels", detections.get("labels", []))

        results_out.append({
            "frame_id": frame_id,
            "boxes": boxes,
            "classes": classes,
            "latency_s": latency,
        })

    print(f"[florence2] Inferencia completada sobre {len(image_paths)} frames")
    return results_out
