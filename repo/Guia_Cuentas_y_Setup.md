# Guía de Cuentas y Setup

Pasos para dejar el proyecto corriendo en Google Colab con GPU T4.

## 1. Google Colab + GPU T4
1. Abre el notebook `notebooks/main_pipeline.ipynb` en Google Colab.
2. Ve a `Entorno de ejecución > Cambiar tipo de entorno de ejecución`.
3. Selecciona **GPU** y el tipo **T4**.

## 2. Roboflow (para descargar CSS-Data)
1. Crea una cuenta gratis en https://roboflow.com
2. Ve a tu perfil > `Settings > API Key` y copia tu **Private API Key**.
3. En Colab, antes de correr el pipeline:
   ```python
   import os
   os.environ["ROBOFLOW_API_KEY"] = "TU_API_KEY_AQUI"
   ```
   O edita directamente `src/config.py` (no recomendado si subes el repo a GitHub).

## 3. Hugging Face (para Florence-2 y Moondream2)
1. Crea una cuenta en https://huggingface.co
2. Genera un token en `Settings > Access Tokens` (permisos de lectura bastan).
3. En Colab:
   ```python
   from huggingface_hub import login
   login("TU_HF_TOKEN_AQUI")
   ```
   Ambos modelos (Florence-2, Moondream2) son públicos, por lo que el login
   es solo para evitar límites de descarga.

## 4. (Opcional) Video real de monitoreo
Si además de las imágenes del split `test` quieren usar un clip `.mp4` real:
- Descarguen un clip libre de derechos desde Pexels o Pixabay (obra en
  construcción, cámara fija).
- Usen `src/data/frame_sampler.py::sample_frames_from_video` para extraer
  frames a ~1 FPS.

## 5. Orden de ejecución recomendado
1. Instalar dependencias (`requirements.txt`).
2. Descargar CSS-Data (`src/data/download.py`).
3. Entrenar YOLOv8n (`src/models/yolo_closed.py`).
4. Correr YOLO-World, Florence-2 y Moondream2 sobre los mismos frames del test.
5. Calcular métricas (`src/metrics/`).
6. Generar tablas y gráficos (`src/utils/visualization.py`).
7. Completar el `README.md` con la respuesta a la pregunta de despliegue.
