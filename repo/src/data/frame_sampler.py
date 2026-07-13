"""
frame_sampler.py
El split "test" de CSS-Data hace de feed de monitoreo: cada imagen es un frame
con ground truth. Este modulo arma la lista de frames (rutas de imagen + label)
que van a evaluar los tres detectores, garantizando que TODOS reciben
exactamente los mismos frames (comparacion justa).

Tambien simula el muestreo ~1 FPS que se justifica en el README: un VLM
generativo no corre a 30 FPS en una T4, por eso se evalua a una tasa reducida
si el usuario decide usar un video real en vez de las imagenes estaticas.
"""

import glob
import os
from dataclasses import dataclass
from typing import List


@dataclass
class Frame:
    image_path: str
    label_path: str
    frame_id: str


def load_test_frames(dataset_dir: str, images_subdir: str = "test/images",
                      labels_subdir: str = "test/labels") -> List[Frame]:
    """
    Carga todos los frames del split test (imagen + label YOLO .txt).
    Se ordenan alfabeticamente para asegurar el mismo orden entre corridas
    y entre los tres modelos.
    """
    images_dir = os.path.join(dataset_dir, images_subdir)
    labels_dir = os.path.join(dataset_dir, labels_subdir)

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) +
                          glob.glob(os.path.join(images_dir, "*.png")))

    frames = []
    for img_path in image_paths:
        frame_id = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, frame_id + ".txt")
        if not os.path.exists(label_path):
            # Frame sin objetos anotados -> label vacia
            label_path = None
        frames.append(Frame(image_path=img_path, label_path=label_path, frame_id=frame_id))

    print(f"[frame_sampler] {len(frames)} frames cargados desde {images_dir}")
    return frames


def sample_frames_from_video(video_path: str, output_dir: str, target_fps: int = 1) -> List[str]:
    """
    (Opcional) Extrae frames de un video .mp4 real a una tasa reducida (target_fps).
    Justificacion: un VLM generativo (Moondream2/Florence-2) no procesa video a
    30 FPS en una T4, por lo que se decide explicitamente reducir la tasa de
    muestreo y agregar resultados en el tiempo.
    """
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(native_fps / target_fps)), 1)

    saved_paths = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"[frame_sampler] {len(saved_paths)} frames extraidos a ~{target_fps} FPS "
          f"(video nativo: {native_fps:.1f} FPS)")
    return saved_paths
