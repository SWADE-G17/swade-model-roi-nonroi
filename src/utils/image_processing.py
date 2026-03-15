"""
utils/image_processing.py

Funciones de procesamiento de imagenes MRI 3D.
Extraidas del notebook original del paper:
"Enhanced ROI guided deep learning model for Alzheimer's detection using 3D MRI images"

Incluye:
- Seleccion de ROIs (apply_mask)
- Mejora de contraste CLAHE (enhance_image)
- Enfoque de imagen (sharpen_image)
- Augmentacion por rotacion (augment)
- Registro no lineal (apply_nonlinear_registration) - definido pero no usado en entrenamiento
"""

import numpy as np
import cv2
from skimage.filters import unsharp_mask
from scipy.ndimage import rotate


# Labels de FreeSurfer para las 6 ROIs del paper:
# 17, 53 = hipocampo izq/der
# 2,  41 = sustancia blanca cerebral izq/der
# 7,  46 = sustancia blanca cerebelar izq/der
ROI_LABELS = [17, 53, 2, 7, 41, 46]


def apply_mask(aseg_image, brain_image, labels=None):
    """
    Enmascara el volumen MRI original para conservar solo las ROIs especificadas.

    Args:
        aseg_image: imagen de segmentacion de FastSurfer (aparc.DKTatlas+aseg.deep.mgz)
        brain_image: imagen MRI original (orig.mgz)
        labels: lista de etiquetas FreeSurfer a conservar. Default: 6 ROIs del paper.

    Returns:
        numpy array 3D con el volumen MRI enmascarado
    """
    if labels is None:
        labels = ROI_LABELS

    aseg_data = aseg_image.get_fdata()
    origin_data = brain_image.get_fdata()

    brain_mask = np.zeros_like(aseg_data)
    for label in labels:
        brain_mask += np.where(aseg_data == label, 1, 0)

    return origin_data * brain_mask


def enhance_slice(slice_data):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) a una sola capa 2D.

    Args:
        slice_data: array 2D (un corte axial del volumen)

    Returns:
        array 2D con contraste mejorado
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(slice_data.astype(np.uint8))


def enhance_image(img_data):
    """
    Aplica CLAHE capa por capa en el eje Z del volumen 3D.

    Args:
        img_data: array 3D (volumen MRI)

    Returns:
        array 3D con contraste mejorado en cada capa
    """
    enhanced = np.zeros_like(img_data)
    for i in range(img_data.shape[2]):
        enhanced[:, :, i] = enhance_slice(img_data[:, :, i])
    return enhanced


def sharpen_image(image, strength=1.0):
    """
    Aplica enfoque (unsharp mask) al volumen 3D.

    Args:
        image: array 3D (volumen MRI)
        strength: intensidad del enfoque (default 1.0)

    Returns:
        array 3D con detalles mejorados
    """
    return unsharp_mask(image, radius=1, amount=strength)


def augment(image, rotation_range=50):
    """
    Aplica una rotacion aleatoria al volumen 3D (augmentacion de datos).
    Solo se usa durante el entrenamiento, no durante la inferencia.

    Args:
        image: array 3D (volumen MRI)
        rotation_range: angulo maximo de rotacion en grados (default 50)

    Returns:
        array 3D rotado aleatoriamente
    """
    rotation_angle = np.random.uniform(-rotation_range, rotation_range)
    return rotate(image, rotation_angle, reshape=False)


def apply_nonlinear_registration(moving_image, fixed_image):
    """
    Aplica registro no lineal (Symmetric Diffeomorphic Registration) entre dos volumenes.
    Definida en el paper pero no utilizada activamente en el pipeline de entrenamiento.

    Args:
        moving_image: volumen a alinear
        fixed_image: volumen de referencia

    Returns:
        moving_image alineado al espacio de fixed_image
    """
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric

    metric = CCMetric(3)
    sdr = SymmetricDiffeomorphicRegistration(
        metric, [10, 10, 10], step_length=0.25, ss_sigma_factor=1.5
    )
    mapping = sdr.optimize(fixed_image, moving_image)
    return mapping.transform(moving_image)
