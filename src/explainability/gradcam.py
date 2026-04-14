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
    from explainability.gradcam import compute_gradcam_3d, visualize_gradcam_slices

    heatmap = compute_gradcam_3d(model, image_array, class_idx=0)
    visualize_gradcam_slices(image_array, heatmap, save_path="gradcam.png")

REFERENCIAS:
    - Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"
    - Adaptado a 3D para volumenes MRI
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf


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
        if isinstance(layer, tf.keras.layers.Conv3D):
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

    # Modelo que da los feature maps de la capa conv + la prediccion final
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array, training=False)

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

    plt.show()
    return fig


def run_gradcam_on_subject(model_path, aseg_path, orig_path,
                            class_names=None, save_path=None,
                            gradcam_class_idx=None):
    """
    Funcion de alto nivel: carga el modelo, preprocesa la imagen,
    calcula Grad-CAM y visualiza el resultado.

    Args:
        model_path: ruta al archivo .h5 del modelo entrenado
        aseg_path: ruta al aparc.DKTatlas+aseg.deep.mgz
        orig_path: ruta al orig.mgz
        class_names: lista con nombres de clases, ej: ["AD", "MCI"]
        save_path: si se especifica, guarda la figura
        gradcam_class_idx: indice de clase para el mapa (ej. 1 en modelos OvR
            para la clase positiva). None = usar la clase predicha (argmax).

    Returns:
        heatmap, predicted_class, probabilities
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from inference.model_loading import load_model_for_inference

    import nibabel
    from skimage.transform import resize
    from utils.image_processing import apply_mask, enhance_image, sharpen_image

    TARGET_SHAPE = (100, 100, 100)

    print("Cargando modelo...")
    model = load_model_for_inference(model_path, target_shape=TARGET_SHAPE, compile=False)

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
    heatmap, predicted_class, probabilities = compute_gradcam_3d(
        model, image_batch, class_idx=gradcam_class_idx
    )

    if class_names is None:
        class_names = [f"Clase {i}" for i in range(len(probabilities))]

    display_class = (
        gradcam_class_idx if gradcam_class_idx is not None else predicted_class
    )
    display_class = int(np.clip(display_class, 0, len(class_names) - 1))

    print(
        f"\nGrad-CAM (clase explicada): {class_names[display_class]} "
        f"({probabilities[display_class]*100:.1f}%)"
    )
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {name}: {prob*100:.1f}%")

    visualize_gradcam_slices(
        image_batch, heatmap,
        class_names=class_names,
        predicted_class=display_class,
        probabilities=probabilities,
        save_path=save_path,
    )

    return heatmap, predicted_class, probabilities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grad-CAM 3D para diagnostico de Alzheimer")
    parser.add_argument("--model", required=True, help="Ruta al modelo .h5")
    parser.add_argument("--aseg", required=True, help="Ruta al aparc.DKTatlas+aseg.deep.mgz")
    parser.add_argument("--orig", required=True, help="Ruta al orig.mgz")
    parser.add_argument("--classes", default="AD,MCI", help="Nombres de clases separados por coma")
    parser.add_argument("--save", default=None, help="Ruta para guardar la figura")
    args = parser.parse_args()

    class_names = args.classes.split(",")
    run_gradcam_on_subject(args.model, args.aseg, args.orig, class_names, args.save)
