"""
explainability/gradcam.py

Implementacion de Grad-CAM 3D para visualizar que regiones del cerebro
activa el modelo al hacer su diagnostico.

Grad-CAM (Gradient-weighted Class Activation Mapping) calcula el gradiente
de la clase predicha respecto a los feature maps de la ultima capa
convolucional. Esas regiones con gradientes altos son las que el modelo
considera importantes para su decision.

En el contexto del paper, esto permite verificar que el modelo
efectivamente se enfoca en las 6 ROIs (hipocampo, sustancia blanca)
y no en regiones irrelevantes.

USO:
    from explainability.gradcam import compute_gradcam_3d, visualize_gradcam_slices, save_gradcam_volume

    heatmap, pred, probs = compute_gradcam_3d(model, image_array, class_idx=0)
    visualize_gradcam_slices(image_array, heatmap, save_path="gradcam.png")
    save_gradcam_volume(heatmap, "gradcam.nii.gz", reference_img="orig.mgz")

REFERENCIAS:
    - Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"
    - Adaptado a 3D para volumenes MRI
"""

from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

keras: Any = tf.keras  # pyright: ignore[reportAttributeAccessIssue]


def get_last_conv_layer_name(model):
    """
    Encuentra automaticamente el nombre de la ultima capa Conv3D del modelo.

    Args:
        model: modelo Keras cargado

    Returns:
        nombre de la ultima capa Conv3D
    """
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv3D):
            last_conv = layer.name
    if last_conv is None:
        raise ValueError("No se encontro ninguna capa Conv3D en el modelo.")
    return last_conv


def compute_gradcam_3d(model, image_array, class_idx=None, conv_layer_name=None):
    """
    Calcula el mapa de activacion Grad-CAM 3D para una imagen.

    Args:
        model: modelo Keras cargado (.h5)
        image_array: numpy array de forma (1, D, H, W, 1) - imagen preprocesada
        class_idx: indice de la clase para la que calcular Grad-CAM.
                   Si es None, usa la clase predicha por el modelo.
        conv_layer_name: nombre de la capa convolucional a usar.
                         Si es None, usa la ultima Conv3D automaticamente.

    Returns:
        heatmap: numpy array 3D (D, H, W) con el mapa de calor normalizado [0, 1]
        predicted_class: indice de la clase predicha
        probabilities: array con las probabilidades de cada clase
    """
    if conv_layer_name is None:
        conv_layer_name = get_last_conv_layer_name(model)

    # En Keras 3, `model.inputs` siempre es una lista. Si es un solo tensor,
    # lo desempacamos para que el grad_model espere un tensor crudo y no una
    # estructura tipo lista/dict (evita el UserWarning de "structure of inputs").
    model_inputs = model.inputs[0] if len(model.inputs) == 1 else model.inputs
    model_output = model.outputs[0] if len(model.outputs) == 1 else model.output

    # Modelo que da los feature maps de la capa conv + la prediccion final
    grad_model = keras.models.Model(
        inputs=model_inputs,
        outputs=[model.get_layer(conv_layer_name).output, model_output],
    )

    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        outputs = grad_model(image_tensor, training=False)

        # Keras 3 a veces envuelve cada salida en una lista; desempaquetar
        # defensivamente para que `predictions` sea un tensor 2D (batch, classes).
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            conv_outputs, predictions = outputs
        else:
            raise RuntimeError(
                f"grad_model devolvio una estructura inesperada: {type(outputs)}"
            )
        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        probabilities = predictions[0].numpy()
        predicted_class = int(np.argmax(probabilities))

        if class_idx is None:
            class_idx = predicted_class

        # Gradiente de la clase seleccionada respecto a los feature maps
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # Promedio global de los gradientes por canal (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    # Ponderar los feature maps por la importancia de cada canal
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # ReLU: solo nos interesan las activaciones positivas
    heatmap = np.maximum(heatmap, 0)

    # Normalizar a [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap, predicted_class, probabilities


def resize_heatmap_3d(heatmap, target_shape):
    """
    Redimensiona el heatmap 3D al tamano original de la imagen.

    Args:
        heatmap: array 3D con el mapa de calor
        target_shape: tupla (D, H, W) con el tamano objetivo

    Returns:
        heatmap redimensionado
    """
    from skimage.transform import resize
    return resize(heatmap, target_shape, anti_aliasing=True)


def visualize_gradcam_slices(image_array, heatmap, class_names=None,
                              predicted_class=None, probabilities=None,
                              save_path=None, num_slices=5):
    """
    Visualiza el Grad-CAM superpuesto sobre la imagen MRI en cortes axiales.

    Args:
        image_array: numpy array (D, H, W) con la imagen MRI
        heatmap: numpy array (D, H, W) con el mapa de calor Grad-CAM
        class_names: lista con los nombres de las clases, ej: ["AD", "MCI"]
        predicted_class: indice de la clase predicha
        probabilities: array con las probabilidades
        save_path: si se especifica, guarda la figura en esa ruta
        num_slices: numero de cortes axiales a mostrar
    """
    # Asegurar que la imagen sea 3D
    if image_array.ndim == 5:
        image_array = image_array[0, :, :, :, 0]
    elif image_array.ndim == 4:
        image_array = image_array[0, :, :, :] if image_array.shape[0] == 1 else image_array[:, :, :, 0]

    # Redimensionar heatmap al tamano de la imagen
    if heatmap.shape != image_array.shape:
        heatmap_resized = resize_heatmap_3d(heatmap, image_array.shape)
    else:
        heatmap_resized = heatmap

    # Seleccionar cortes axiales equidistantes en el centro del volumen
    depth = image_array.shape[2]
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, num_slices, dtype=int)

    fig, axes = plt.subplots(2, num_slices, figsize=(num_slices * 3, 7))

    for i, slice_idx in enumerate(slice_indices):
        img_slice = image_array[:, :, slice_idx]
        heat_slice = heatmap_resized[:, :, slice_idx]

        # Normalizar imagen para visualizacion
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        # Fila superior: imagen original
        axes[0, i].imshow(img_norm, cmap="gray")
        axes[0, i].set_title(f"Corte {slice_idx}", fontsize=9)
        axes[0, i].axis("off")

        # Fila inferior: Grad-CAM superpuesto
        axes[1, i].imshow(img_norm, cmap="gray")
        axes[1, i].imshow(heat_slice, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        axes[1, i].set_title("Grad-CAM", fontsize=9)
        axes[1, i].axis("off")

    # Titulo con el diagnostico
    title = "Grad-CAM 3D - Activaciones del modelo"
    if predicted_class is not None and class_names is not None and probabilities is not None:
        class_name = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)
        prob = probabilities[predicted_class] * 100
        title += f"\nDiagnostico: {class_name} ({prob:.1f}%)"

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM guardado en: {save_path}")

    plt.close(fig)
    return fig


def save_gradcam_volume(heatmap, output_path, reference_img=None, affine=None):
    """
    Guarda el mapa Grad-CAM 3D como volumen NIfTI (.nii, .nii.gz) o FreeSurfer (.mgz, .mgh).

    Args:
        heatmap: array 3D (D, H, W), valores tipicamente en [0, 1] tras compute_gradcam_3d.
        output_path: ruta de salida; la extension determina el formato (.nii.gz, .nii, .mgz, .mgh).
        reference_img: opcional. Ruta a un volumen nibabel o objeto Nifti1Image/MGHImage.
                       Si se pasa, se copia el affine y, si las dimensiones no coinciden,
                       se reescala el heatmap a la rejilla del volumen de referencia.
        affine: matriz 4x4 opcional. Solo se usa si reference_img es None.
                  Si ambos son None, se usa np.eye(4) (espacio anonimo, 1 mm voxels).

    Returns:
        La ruta output_path.
    """
    import os

    import nibabel as nib
    from nibabel.freesurfer.mghformat import MGHImage
    from nibabel.spatialimages import SpatialImage

    path_lower = output_path.lower()
    if path_lower.endswith(".nii.gz") or path_lower.endswith(".nii"):
        fmt = "nifti"
    elif path_lower.endswith(".mgz") or path_lower.endswith(".mgh"):
        fmt = "mgh"
    else:
        raise ValueError(
            "Extension no soportada. Use .nii, .nii.gz, .mgz o .mgh; recibido: "
            + os.path.basename(output_path)
        )

    h = np.asarray(heatmap, dtype=np.float32)
    if h.ndim != 3:
        raise ValueError(f"heatmap debe ser 3D (D,H,W); forma recibida: {h.shape}")

    ref_affine = np.eye(4, dtype=np.float64)
    if reference_img is not None:
        if isinstance(reference_img, str):
            reference_img = nib.load(reference_img)
        ref_img = cast(SpatialImage, reference_img)
        ref_img_affine = ref_img.affine
        if ref_img_affine is None:
            raise ValueError(
                "reference_img no tiene un affine valido; no se puede preservar la geometria."
            )
        ref_affine = ref_img_affine.copy()
        ref_data = np.asarray(ref_img.dataobj)
        if ref_data.ndim == 4 and ref_data.shape[-1] == 1:
            ref_data = ref_data[..., 0]
        if ref_data.ndim != 3:
            raise ValueError(
                "reference_img debe ser un volumen 3D o 4D con un solo canal en el ultimo eje; "
                f"forma: {np.asarray(ref_img.dataobj).shape}"
            )
        ref_shape = ref_data.shape
        if h.shape != ref_shape:
            h = np.asarray(resize_heatmap_3d(h, ref_shape), dtype=np.float32)
    elif affine is not None:
        ref_affine = np.asarray(affine, dtype=np.float64)
        if ref_affine.shape != (4, 4):
            raise ValueError("affine debe ser una matriz 4x4.")

    data_for_img = cast(Any, h)
    if fmt == "nifti":
        out_img = nib.Nifti1Image(data_for_img, ref_affine)
    else:
        out_img = MGHImage(data_for_img, ref_affine)

    nib.save(out_img, output_path)
    print(f"Volumen Grad-CAM guardado en: {output_path}")
    return output_path


def run_gradcam_on_subject(model_path, aseg_path, orig_path,
                            class_names=None, save_path=None, volume_path=None):
    """
    Funcion de alto nivel: carga el modelo, preprocesa la imagen,
    calcula Grad-CAM y visualiza el resultado.

    Args:
        model_path: ruta al archivo .h5 del modelo entrenado
        aseg_path: ruta al aparc.DKTatlas+aseg.deep.mgz
        orig_path: ruta al orig.mgz
        class_names: lista con nombres de clases, ej: ["AD", "MCI"]
        save_path: si se especifica, guarda la figura PNG
        volume_path: si se especifica, guarda el mapa 3D como .nii.gz, .nii, .mgz o .mgh,
                      reescalado a la rejilla de orig.mgz para poder superponerlo en viewers.

    Returns:
        heatmap, predicted_class, probabilities
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import nibabel
    from skimage.transform import resize
    from utils.image_processing import apply_mask, enhance_image, sharpen_image

    TARGET_SHAPE = (100, 100, 100)

    print("Cargando modelo...")
    model = keras.models.load_model(model_path, compile=False)

    print("Preprocesando imagen...")
    aseg_image = nibabel.load(aseg_path)
    orig_image = nibabel.load(orig_path)

    image = apply_mask(aseg_image, orig_image)
    image = resize(image, TARGET_SHAPE, anti_aliasing=True)
    image = enhance_image(image)
    image = sharpen_image(image)

    image_batch = np.expand_dims(image, axis=0)
    image_batch = np.expand_dims(image_batch, axis=-1)

    print("Calculando Grad-CAM...")
    heatmap, predicted_class, probabilities = compute_gradcam_3d(model, image_batch)

    if class_names is None:
        class_names = [f"Clase {i}" for i in range(len(probabilities))]

    print(f"\nDiagnostico: {class_names[predicted_class]} ({probabilities[predicted_class]*100:.1f}%)")
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {name}: {prob*100:.1f}%")

    if save_path:
        visualize_gradcam_slices(
            image_batch, heatmap,
            class_names=class_names,
            predicted_class=predicted_class,
            probabilities=probabilities,
            save_path=save_path,
        )

    if volume_path:
        save_gradcam_volume(heatmap, volume_path, reference_img=orig_image)

    return heatmap, predicted_class, probabilities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grad-CAM 3D para diagnostico de Alzheimer")
    parser.add_argument("--model", required=True, help="Ruta al modelo .h5")
    parser.add_argument("--aseg", required=True, help="Ruta al aparc.DKTatlas+aseg.deep.mgz")
    parser.add_argument("--orig", required=True, help="Ruta al orig.mgz")
    parser.add_argument("--classes", default="AD,MCI", help="Nombres de clases separados por coma")
    parser.add_argument("--save", default=None, help="Ruta para guardar la figura")
    parser.add_argument(
        "--volume",
        default=None,
        help="Ruta para guardar el mapa Grad-CAM 3D (.nii.gz, .nii, .mgz, .mgh)",
    )
    args = parser.parse_args()

    class_names = args.classes.split(",")
    run_gradcam_on_subject(
        args.model,
        args.aseg,
        args.orig,
        class_names,
        save_path=args.save,
        volume_path=args.volume,
    )
