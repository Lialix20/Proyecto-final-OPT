"""
download.py
Descarga automatica del dataset CSS-Data (Construction Site Safety Image Dataset)
desde Roboflow Universe, en formato YOLOv8 (train/valid/test).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    DATA_DIR, ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT, ROBOFLOW_VERSION,
)


def download_css_data(target_dir: str = DATA_DIR) -> str:
    """
    Descarga CSS-Data desde Roboflow en formato YOLOv8.
    Devuelve la ruta a la carpeta del dataset descargado (contiene data.yaml).
    """
    if ROBOFLOW_API_KEY == "TU_API_KEY_AQUI":
        raise ValueError(
            "Debes configurar tu ROBOFLOW_API_KEY. "
            "Ver Guia_Cuentas_y_Setup.md o exportar la variable de entorno "
            "ROBOFLOW_API_KEY antes de ejecutar este script."
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download("yolov8", location=target_dir)

    print(f"[download] Dataset descargado en: {dataset.location}")
    return dataset.location


if __name__ == "__main__":
    download_css_data()
