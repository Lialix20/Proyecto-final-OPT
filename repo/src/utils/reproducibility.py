"""
reproducibility.py
Fija semillas en todas las librerias relevantes para maximizar reproducibilidad.
Nota: con GPU (CUDA), algunas operaciones siguen siendo no deterministas
(ver advertencia del enunciado). Este modulo minimiza esa variabilidad.
"""

import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Fija la semilla en random, numpy y torch (CPU y CUDA si esta disponible)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinismo adicional (puede ser mas lento, pero mas reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[reproducibility] Semilla global fijada en {seed}")
