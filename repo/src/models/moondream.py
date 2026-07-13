"""
moondream.py
Paradigma 3 (VLM generativo): Moondream2 (Apache-2.0).
Responde en lenguaje natural la pregunta-evento por frame:
"¿hay alguien sin casco?" -> usado para la metrica F1 nivel-evento.
Tambien se usa para el experimento de robustez / sensibilidad al prompt.
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import MOONDREAM_MODEL_ID, MOONDREAM_REVISION, DEVICE


def load_moondream():
    """Carga Moondream2 y su tokenizer desde HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_MODEL_ID, revision=MOONDREAM_REVISION,
        trust_remote_code=True, torch_dtype=torch.float16,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MOONDREAM_MODEL_ID, revision=MOONDREAM_REVISION)

    print(f"[moondream] Modelo cargado: {MOONDREAM_MODEL_ID}")
    return model, tokenizer


def _parse_yes_no(answer: str) -> bool:
    """Convierte la respuesta libre del VLM en una etiqueta binaria de violacion."""
    answer_lower = answer.lower()
    negative_hints = ["no one", "nobody", "everyone is wearing", "all workers", "yes, everyone"]
    positive_hints = ["yes", "without a hardhat", "not wearing", "missing", "no hardhat", "no helmet"]

    if any(hint in answer_lower for hint in negative_hints):
        return False
    if any(hint in answer_lower for hint in positive_hints):
        return True
    return False  # por defecto, si es ambiguo, se asume "sin violacion detectada"


def predict_moondream(model, tokenizer, image_paths: list, prompt: str):
    """
    Corre la pregunta-evento sobre cada frame con UN prompt fijo.
    Devuelve lista de dicts: {frame_id, raw_answer, violation_predicted, latency_s}.
    """
    from PIL import Image

    results_out = []
    for img_path in image_paths:
        frame_id = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, prompt, tokenizer)
        latency = time.perf_counter() - t0

        results_out.append({
            "frame_id": frame_id,
            "raw_answer": answer,
            "violation_predicted": _parse_yes_no(answer),
            "latency_s": latency,
        })

    print(f"[moondream] Inferencia completada sobre {len(image_paths)} frames (prompt: '{prompt}')")
    return results_out


def run_prompt_robustness(model, tokenizer, image_paths: list, prompt_variants: list):
    """
    Corre el mismo set de frames con distintas variantes de prompt (redaccion),
    para medir la sensibilidad al prompt (robustez).
    Devuelve dict {prompt: [resultados por frame]}.
    """
    all_results = {}
    for prompt in prompt_variants:
        all_results[prompt] = predict_moondream(model, tokenizer, image_paths, prompt)
    return all_results
