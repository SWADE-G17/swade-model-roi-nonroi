"""
reading/data_loader.py

Funciones para cargar y generar los datos de entrenamiento y evaluacion.
Extraidas del notebook original del paper:
"Enhanced ROI guided deep learning model for Alzheimer's detection using 3D MRI images"

Incluye:
- image_loader_roi: carga y preprocesa una sola imagen
- data_generator: generador de batches para el entrenamiento
- load_dataset: escanea las carpetas AD/MCI/CN y retorna las rutas con sus etiquetas
"""

import os
import numpy as np
import nibabel
from skimage.transform import resize
from tensorflow.keras.utils import to_categorical

from utils.image_processing import apply_mask, enhance_image, sharpen_image, augment


# Etiquetas numericas para cada clase
CLASS_LABELS = {
    "AD":  0,
    "MCI": 1,
    "CN":  2,
}


def image_loader_roi(image_path, target_shape, is_training=False):
    """
    Carga y preprocesa una imagen MRI para el modelo ROI.

    Pipeline:
        1. Cargar aseg (segmentacion) y orig (imagen original) desde FastSurfer
        2. Aplicar mascara de ROIs (6 regiones del paper)
        3. Redimensionar al tamano objetivo
        4. CLAHE (mejora de contraste)
        5. Sharpening (enfoque)
        6. Augmentacion aleatoria (solo en entrenamiento)

    Args:
        image_path: ruta al archivo aparc.DKTatlas+aseg.deep.mgz
        target_shape: tupla (D, H, W) con el tamano objetivo, ej: (100, 100, 100)
        is_training: si True, aplica augmentacion por rotacion

    Returns:
        numpy array 3D preprocesado listo para el modelo
    """
    aseg_image = nibabel.load(image_path)

    # orig.mgz esta en la misma carpeta que aseg
    mri_dir = os.path.dirname(image_path)
    orig_path = os.path.join(mri_dir, "orig.mgz")
    orig_image = nibabel.load(orig_path)

    image = apply_mask(aseg_image, orig_image)
    image = resize(image, target_shape, anti_aliasing=True)
    image = enhance_image(image)
    image = sharpen_image(image)

    if is_training:
        image = augment(image, rotation_range=50)

    return image


def data_generator(paths, labels, batch_size, target_shape, is_training=False):
    """
    Generador infinito de batches para el entrenamiento o evaluacion.

    Args:
        paths: array de rutas a los archivos .mgz
        labels: array de etiquetas enteras correspondientes
        batch_size: numero de imagenes por batch
        target_shape: tupla con el tamano objetivo del volumen
        is_training: si True, aplica augmentacion en la carga

    Yields:
        (batch_images, batch_labels): arrays numpy listos para el modelo
    """
    num_classes = len(np.unique(labels))

    while True:
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            batch_images = [
                image_loader_roi(p, target_shape, is_training=is_training)
                for p in batch_paths
            ]

            # Agregar dimension de canal: (batch, D, H, W, 1)
            batch_images = np.stack([batch_images] * 1, axis=-1)
            batch_labels = to_categorical(batch_labels, num_classes=num_classes)

            yield np.array(batch_images), batch_labels


def load_dataset(base_dir, classes=None):
    """
    Escanea las carpetas de datos procesados por FastSurfer y retorna
    las rutas a los archivos .mgz con sus etiquetas correspondientes.

    Estructura esperada:
        base_dir/
            AD/
                subj001/mri/aparc.DKTatlas+aseg.deep.mgz
                subj001/mri/orig.mgz
                ...
            MCI/
                ...
            CN/
                ...

    Args:
        base_dir: ruta a la carpeta raiz con las subcarpetas AD/, MCI/, CN/
        classes: lista de clases a cargar. Default: todas las disponibles.

    Returns:
        image_paths: lista de rutas a los archivos aseg.mgz
        labels: lista de etiquetas enteras
        class_counts: dict con el conteo por clase
    """
    if classes is None:
        classes = [c for c in CLASS_LABELS if os.path.exists(os.path.join(base_dir, c))]

    image_paths = []
    labels = []
    class_counts = {}

    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"ADVERTENCIA: No existe la carpeta {class_dir}")
            continue

        label = CLASS_LABELS[class_name]
        count = 0

        for subject_dir in sorted(os.listdir(class_dir)):
            subject_path = os.path.join(class_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            mri_dir = os.path.join(subject_path, "mri")
            aseg_path = os.path.join(mri_dir, "aparc.DKTatlas+aseg.deep.mgz")
            orig_path = os.path.join(mri_dir, "orig.mgz")

            if os.path.exists(aseg_path) and os.path.exists(orig_path):
                image_paths.append(aseg_path)
                labels.append(label)
                count += 1

        class_counts[class_name] = count
        print(f"  {class_name}: {count} sujetos cargados")

    return image_paths, labels, class_counts


def prepare_ovr_dataset(base_dir, positive_class, random_seed=42):
    """
    Prepara un dataset One-vs-Rest para una clase positiva.

    La clase positiva recibe etiqueta 1.
    Las otras dos clases combinadas reciben etiqueta 0 (el 'rest').
    Las clases se balancean: positivo vs rest con oversampling si es necesario.

    Ejemplo: positive_class="AD"
        positivo: AD (label=1)
        negativo: MCI + CN juntos (label=0)

    Args:
        base_dir: ruta raiz con subcarpetas AD/, MCI/, CN/
        positive_class: una de "AD", "MCI" o "CN"
        random_seed: semilla para reproducibilidad

    Returns:
        image_paths: lista de rutas a los archivos aseg.mgz
        labels: lista de etiquetas (1=positivo, 0=rest)
        info: dict con conteos por clase para logging
    """
    all_classes = ["AD", "MCI", "CN"]
    rest_classes = [c for c in all_classes if c != positive_class]

    def scan_class(class_name):
        class_dir = os.path.join(base_dir, class_name)
        paths = []
        if not os.path.exists(class_dir):
            print(f"ADVERTENCIA: No existe la carpeta {class_dir}")
            return paths
        for subject_dir in sorted(os.listdir(class_dir)):
            subject_path = os.path.join(class_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            mri_dir = os.path.join(subject_path, "mri")
            aseg = os.path.join(mri_dir, "aparc.DKTatlas+aseg.deep.mgz")
            orig = os.path.join(mri_dir, "orig.mgz")
            if os.path.exists(aseg) and os.path.exists(orig):
                paths.append(aseg)
        return paths

    positive_paths = scan_class(positive_class)
    rest_paths = []
    for rc in rest_classes:
        rest_paths += scan_class(rc)

    info = {
        positive_class: len(positive_paths),
        "rest (" + "+".join(rest_classes) + ")": len(rest_paths),
    }

    print(f"\n  OvR [{positive_class} vs rest]:")
    for k, v in info.items():
        print(f"    {k}: {v} sujetos")

    # Balancear: positivo=1, rest=0
    image_paths, labels = balance_binary_dataset(
        positive_paths, rest_paths, label_a=1, label_b=0
    )
    return image_paths, labels, info


def balance_binary_dataset(paths_a, paths_b, label_a=0, label_b=1):
    """
    Balancea dos clases por oversampling de la clase minoritaria.
    Replica exactamente la logica del notebook original para AD vs MCI.

    Args:
        paths_a: lista de rutas de la clase A (ej: AD)
        paths_b: lista de rutas de la clase B (ej: MCI)
        label_a: etiqueta para clase A
        label_b: etiqueta para clase B

    Returns:
        image_paths: lista balanceada de rutas
        labels: lista balanceada de etiquetas
    """
    diff = len(paths_a) - len(paths_b)

    if diff > 0:
        # paths_a es mayor, oversample paths_b
        extra = paths_b[:diff]
        image_paths = paths_a + paths_b + extra
        labels = ([label_a] * len(paths_a) +
                  [label_b] * len(paths_b) +
                  [label_b] * len(extra))
    elif diff < 0:
        # paths_b es mayor, oversample paths_a
        extra = paths_a[:abs(diff)]
        image_paths = paths_a + extra + paths_b
        labels = ([label_a] * len(paths_a) +
                  [label_a] * len(extra) +
                  [label_b] * len(paths_b))
    else:
        image_paths = paths_a + paths_b
        labels = [label_a] * len(paths_a) + [label_b] * len(paths_b)

    return image_paths, labels
