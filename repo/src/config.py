"""
config.py
Configuracion global del proyecto: rutas, semillas y constantes compartidas
por los tres detectores (YOLOv8n, YOLO-World, Florence-2 + Moondream2).
"""

import os

# --------------------------------------------------------------------------
# Semilla global (reproducibilidad)
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")            # dataset CSS-Data descargado
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")

for d in [DATA_DIR, RESULTS_DIR, METRICS_DIR, TABLES_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------
# CSS-Data (Construction Site Safety Image Dataset) - CC BY 4.0
# --------------------------------------------------------------------------
# Reemplazar con tu propia API key de Roboflow (ver Guia_Cuentas_y_Setup.md)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "TU_API_KEY_AQUI")
ROBOFLOW_WORKSPACE = "roboflow-universe-projects"
ROBOFLOW_PROJECT = "construction-site-safety"
ROBOFLOW_VERSION = 27  # version publica mas usada del dataset CSS-Data

# 10 clases originales de CSS-Data
CSS_CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]

# Clase de "violacion" para la metrica de nivel-evento (negacion)
VIOLATION_CLASS = "NO-Hardhat"
# Clases compartibles usadas para mAP/IoU entre paradigmas (caja)
SHARED_BOX_CLASSES = ["Person", "Hardhat"]

# Prompts de vocabulario abierto para YOLO-World
YOLO_WORLD_PROMPTS = ["hard hat", "head", "person"]

# Preguntas para el VLM generativo (Moondream2) - nivel evento
# Variantes usadas en el experimento de robustez (sensibilidad al prompt)
MOONDREAM_PROMPT_VARIANTS = [
    "Is there any person in this image without a hardhat?",
    "Does everyone in the picture wear a helmet?",
    "Look at the workers in this photo. Is anyone missing a hard hat?",
    "Are all people wearing head protection in this image?",
    "Identify if there's a worker with no safety helmet in the scene.",
]

# --------------------------------------------------------------------------
# Muestreo de frames (VLM generativo no corre a 30 FPS en T4)
# --------------------------------------------------------------------------
SAMPLING_FPS = 1  # justificacion: ver README.md, seccion "Diseno del muestreo"

# --------------------------------------------------------------------------
# Entrenamiento YOLOv8n
# --------------------------------------------------------------------------
YOLO_EPOCHS = 50
YOLO_IMG_SIZE = 640
YOLO_BATCH = 16
YOLO_MODEL_NAME = "yolov8n.pt"

# --------------------------------------------------------------------------
# Modelos HuggingFace
# --------------------------------------------------------------------------
FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"
MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2024-08-26"

# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------
DEVICE = "cuda"  # se fuerza a cuda; en Colab activar GPU T4 antes de correr
