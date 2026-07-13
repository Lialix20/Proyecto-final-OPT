"""
speed_metrics.py
Metrica de velocidad: FPS y latencia por frame de cada paradigma,
medidos en el mismo hardware (GPU T4 en Colab).
"""

import numpy as np


def compute_speed_stats(latencies_s: list) -> dict:
    """
    latencies_s: lista de latencias por frame en segundos (incluye pre/post-proceso,
    ya medida dentro de cada modulo de prediccion).
    """
    latencies = np.array(latencies_s)
    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = 1.0 / mean_latency if mean_latency > 0 else 0.0

    return {
        "mean_latency_s": mean_latency,
        "std_latency_s": std_latency,
        "fps": fps,
        "n_frames": len(latencies),
    }
