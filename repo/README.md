# ¿Qué detector despliegas, y por qué?

Proyecto Final — Procesamiento de Imágenes Digitales y Visión por Computador (UAH).
Comparación de tres paradigmas de detección para monitoreo de EPP en obras de
construcción: **YOLOv8n** (cerrado), **YOLO-World** (vocabulario abierto) y
**Florence-2 + Moondream2** (VLM).

## Contenido del repositorio

```
requirements.txt          # dependencias
Guia_Cuentas_y_Setup.md   # cómo crear cuentas (Roboflow, HuggingFace) y activar T4
notebooks/main_pipeline.ipynb   # notebook orquestador, ejecutable en Colab T4
src/
  config.py               # semillas, paths, clases, prompts
  data/                    # descarga de CSS-Data y muestreo de frames
  models/                  # yolo_closed, yolo_world, florence2, moondream
  metrics/                 # box_metrics (mAP/IoU), event_metrics (F1),
                           # speed_metrics (FPS), robustness (sensibilidad prompt)
  utils/                   # reproducibility.py, visualization.py
results/                   # metrics/, tables/, plots/, checkpoints/ (se llenan al correr)
```

## Cómo correr

1. Sube este repo a GitHub y ábrelo en Colab con `notebooks/main_pipeline.ipynb`.
2. Activa GPU T4 (ver `Guia_Cuentas_y_Setup.md`).
3. Configura tu `ROBOFLOW_API_KEY`.
4. Ejecuta las celdas en orden. Los resultados se guardan automáticamente en `results/`.

## Qué se implementó

- **Detector cerrado (YOLOv8n)**: fine-tuning sobre CSS-Data (10 clases, incluyendo
  `NO-Hardhat`), evaluado sobre el split `test`.
- **Vocabulario abierto (YOLO-World)**: zero-shot con `set_classes(["hard hat","head","person"])`,
  sin reentrenar.
- **VLM de grounding (Florence-2)**: `<OPEN_VOCABULARY_DETECTION>` con prompt de texto.
- **VLM generativo (Moondream2)**: responde en lenguaje natural la pregunta-evento
  por frame ("¿hay alguien sin casco?"), con 5 variantes de prompt para medir
  sensibilidad a la redacción.
- **Muestreo de frames** a ~1 FPS (el split `test` actúa como feed de monitoreo,
  cada imagen = 1 frame con ground truth), justificado porque un VLM generativo
  no procesa 30 FPS en una T4.
- **Métricas**: mAP@0.5/IoU (nivel-caja, clases compartibles Person/Hardhat), F1
  nivel-evento (los tres paradigmas, sobre la pregunta binaria "¿sin casco?"),
  FPS/latencia (mismo hardware), y robustez (sensibilidad al prompt + clase no vista).
- **Tablas y gráficos** generados automáticamente en `results/tables` y `results/plots`.

## Cuadro comparativo final

*(se completa automáticamente en `results/tables/comparacion_final.csv` al correr
el notebook; pegar aquí los números reales antes de la entrega)*

| Paradigma | mAP@0.5 | F1 nivel-evento | FPS (T4) | Licencia |
|---|---|---|---|---|
| YOLOv8n (cerrado) | — | — | — | AGPL-3.0 |
| YOLO-World (vocab. abierto) | — | — | — | GPL-3.0 |
| Florence-2 (VLM grounding) | — | — | — | MIT |
| Moondream2 (VLM generativo) | N/A | — | — | Apache-2.0 |

## Respuesta a la pregunta final: ¿cuál desplegarían, y por qué?

*(completar con los números obtenidos; ejemplo de estructura de argumento)*

> Recomendamos desplegar **[YOLOv8n / YOLO-World / Florence-2 / Moondream2]**
> porque, sobre los mismos frames de test: (1) su F1 nivel-evento en la pregunta
> "¿sin casco?" fue de **X**, superior a las alternativas; (2) corre a **X FPS**
> en la misma T4, [suficiente / insuficiente] para un feed de cámara de obra;
> (3) en el experimento de robustez, [mostró baja sensibilidad al prompt / falló
> al generalizar a clases no vistas como "guantes"].
>
> **Trade-offs considerados:**
> - Si mañana piden detectar "guantes": [paradigma] permite hacerlo sin reentrenar,
>   a diferencia de YOLOv8n.
> - Si la cámara es de 30 FPS real: solo el paradigma cerrado (o vocabulario
>   abierto) sostiene esa tasa; el VLM generativo requiere muestreo.
> - Si un falso negativo de "sin casco" implica multa: priorizamos **recall**
>   sobre la clase de violación, no solo F1 promedio.
> - Licencia: [AGPL-3.0 / GPL-3.0 / MIT / Apache-2.0] — [compatible / no
>   compatible] con un despliegue comercial cerrado.

## Videos

- Video 1 (los tres sistemas funcionando sobre el mismo feed): `[enlace YouTube no listado]`
- Video 2 (<3 min, justificación de la decisión de despliegue): `[enlace YouTube no listado]`

## Limitaciones y advertencias documentadas

- La comparación de mAP con el VLM generativo no es directa (enumera instancias
  de forma inconsistente); por eso el F1 nivel-evento es la comparación principal.
- Reproducibilidad imperfecta por no-determinismo de CUDA; se guardan todos los
  checkpoints y resultados intermedios por si Colab se desconecta.
- Negación y atributos ("sin casco") son el punto donde vocabulario abierto y
  VLM tienden a fallar — es el hallazgo central del proyecto, no un error de
  implementación.
