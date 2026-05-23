"""
inference/predict.py

Script de diagnostico para una imagen MRI nueva.
Soporta dos modos:

  --mode binary  (default)
      Usa UN solo modelo binario (ej: AD vs MCI).
      Util para comparar pares de clases como en el paper.

  --mode ovr
      Usa TRES modelos One-vs-Rest para dar un diagnostico de 3 clases:
      AD, MCI o CN.
      Cada modelo predice la probabilidad de ser su clase positiva.
      Se elige la clase con mayor probabilidad.

REQUISITOS:
    - Imagen procesada con FastSurfer (genera orig.mgz y aparc.DKTatlas+aseg.deep.mgz)
    - Modelos .h5 entrenados con training/train.py

USO - Modo binario (un modelo):
    python inference/predict.py \
        --subject_dir C:/paciente/mri \
        --model model_ADNI_AD_vs_MCI.h5 \
        --classes AD,MCI

USO - Modo OvR (tres modelos, diagnostico completo AD/MCI/CN):
    python inference/predict.py \
        --subject_dir C:/paciente/mri \
        --mode ovr \
        --model_ad  model_ADNI_AD_vs_rest.h5 \
        --model_mci model_ADNI_MCI_vs_rest.h5 \
        --model_cn  model_ADNI_CN_vs_rest.h5

SALIDA MODO OvR:
    ========================================
    RESULTADO DEL DIAGNOSTICO (One-vs-Rest)
    ========================================
    AD  (Alzheimer)         :  87.3%  ##############################  <-- DIAGNOSTICO
    MCI (Deterioro leve)    :  45.1%  ###############
    CN  (Normal)            :  12.4%  ####

    Diagnostico mas probable: AD (Alzheimer)
    ========================================
    ADVERTENCIA: Resultado orientativo.
    No reemplaza el criterio medico especializado.
    ========================================

NOTA PARA MAC M4:
    pip install tensorflow-macos tensorflow-metal
    El resto del script funciona igual.
"""

import argparse
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_default_model():
    """Return the first .h5 file found in the inference/ directory, or None."""
    candidates = glob.glob(os.path.join(INFERENCE_DIR, "*.h5"))
    return candidates[0] if candidates else None


# ============================================================
# Preprocesado (identico al entrenamiento)
# ============================================================

def preprocess_subject(aseg_path, orig_path, target_shape=(100, 100, 100)):
    """
    Pipeline de preprocesado identico al usado durante el entrenamiento:
    apply_mask -> resize -> CLAHE -> sharpening -> batch/canal dims.

    Returns: numpy array (1, D, H, W, 1)
    """
    import nibabel
    from skimage.transform import resize
    from utils.image_processing import apply_mask, enhance_image, sharpen_image

    aseg_image = nibabel.load(aseg_path)
    orig_image = nibabel.load(orig_path)

    image = apply_mask(aseg_image, orig_image)
    image = resize(image, target_shape, anti_aliasing=True)
    image = enhance_image(image)
    image = sharpen_image(image)

    image = np.expand_dims(image, axis=0)   # batch
    image = np.expand_dims(image, axis=-1)  # canal
    return image


# ============================================================
# Prediccion - Modo binario
# ============================================================

def predict_binary(aseg_path, orig_path, model_path=None,
                   target_shape=(100, 100, 100), class_names=None):
    """
    Prediccion con un solo modelo binario.

    If *model_path* is ``None``, the first ``.h5`` file found in the
    ``inference/`` directory is used automatically.

    Returns:
        dict con predicted_class, predicted_name, probabilities, class_names
    """
    import tensorflow as tf

    if model_path is None:
        model_path = _find_default_model()
        if model_path is None:
            print("\nERROR: No se encontro ningun archivo .h5 en inference/")
            sys.exit(1)

    if class_names is None:
        class_names = ["Clase 0", "Clase 1"]

    print(f"\nCargando modelo: {os.path.basename(model_path)}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"\nERROR al cargar el modelo: {e}")
        sys.exit(1)

    print("Preprocesando imagen...")
    try:
        image = preprocess_subject(aseg_path, orig_path, target_shape)
    except Exception as e:
        print(f"\nERROR al preprocesar la imagen: {e}")
        sys.exit(1)

    print("Realizando prediccion...")
    probabilities = model.predict(image, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))

    return {
        "predicted_class": predicted_class,
        "predicted_name": class_names[predicted_class],
        "probabilities": probabilities,
        "class_names": class_names,
        "mode": "binary",
    }


# ============================================================
# Prediccion - Modo One-vs-Rest (3 modelos)
# ============================================================

def predict_ovr(aseg_path, orig_path,
                model_ad_path, model_mci_path, model_cn_path,
                target_shape=(100, 100, 100)):
    """
    Prediccion usando 3 modelos One-vs-Rest.

    Cada modelo fue entrenado para distinguir su clase positiva del resto:
      - model_ad:  AD=1 vs (MCI+CN)=0  → prob[1] = probabilidad de ser AD
      - model_mci: MCI=1 vs (AD+CN)=0  → prob[1] = probabilidad de ser MCI
      - model_cn:  CN=1 vs (AD+MCI)=0  → prob[1] = probabilidad de ser CN

    El diagnostico final es la clase con mayor probabilidad de ser "positiva".

    Returns:
        dict con predicted_class, predicted_name, probabilities, class_names
    """
    import tensorflow as tf

    print("\nCargando los 3 modelos OvR...")
    models = {}
    for name, path in [("AD", model_ad_path), ("MCI", model_mci_path), ("CN", model_cn_path)]:
        if not os.path.exists(path):
            print(f"\nERROR: No se encontro el modelo {name}: {path}")
            sys.exit(1)
        try:
            models[name] = tf.keras.models.load_model(path, compile=False)
            print(f"  [{name}] cargado: {os.path.basename(path)}")
        except Exception as e:
            print(f"\nERROR al cargar modelo {name}: {e}")
            sys.exit(1)

    print("Preprocesando imagen (una sola vez para los 3 modelos)...")
    try:
        image = preprocess_subject(aseg_path, orig_path, target_shape)
    except Exception as e:
        print(f"\nERROR al preprocesar la imagen: {e}")
        sys.exit(1)

    print("Realizando 3 predicciones...")
    # prob_positive = probabilidad de que pertenezca a esa clase (indice 1)
    ovr_probs = {}
    for name, model in models.items():
        probs = model.predict(image, verbose=0)[0]
        ovr_probs[name] = float(probs[1])  # prob de ser la clase positiva
        print(f"  P({name}) = {ovr_probs[name]*100:.1f}%")

    # Normalizar las 3 probabilidades para que sumen 1 (mas interpretable)
    total = sum(ovr_probs.values())
    if total > 0:
        normalized = {k: v / total for k, v in ovr_probs.items()}
    else:
        normalized = {k: 1/3 for k in ovr_probs}

    class_names = ["AD", "MCI", "CN"]
    probabilities = np.array([normalized["AD"], normalized["MCI"], normalized["CN"]])
    predicted_class = int(np.argmax(probabilities))

    return {
        "predicted_class": predicted_class,
        "predicted_name": class_names[predicted_class],
        "probabilities": probabilities,
        "class_names": class_names,
        "raw_ovr": ovr_probs,
        "mode": "ovr",
    }


# ============================================================
# Presentacion del resultado
# ============================================================

CLASS_DESCRIPTIONS = {
    "AD":  "Alzheimer",
    "MCI": "Deterioro cognitivo leve",
    "CN":  "Normal",
}


def print_result(result):
    """Imprime el resultado del diagnostico de forma clara."""
    class_names = result["class_names"]
    probabilities = result["probabilities"]
    predicted_class = result["predicted_class"]
    predicted_name = result["predicted_name"]
    mode = result.get("mode", "binary")

    desc = CLASS_DESCRIPTIONS.get(predicted_name, predicted_name)

    print()
    header = "RESULTADO DEL DIAGNOSTICO"
    if mode == "ovr":
        header += " (One-vs-Rest)"
    print("=" * 56)
    print(f"  {header}")
    print("=" * 56)

    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        bar = "#" * int(prob * 28)
        marker = "  <-- DIAGNOSTICO" if i == predicted_class else ""
        cls_desc = CLASS_DESCRIPTIONS.get(name, name)
        print(f"  {name} ({cls_desc:28s}): {prob*100:5.1f}%  {bar}{marker}")

    print()
    print(f"  Diagnostico mas probable: {predicted_name} ({desc})")
    print("=" * 56)
    print("  ADVERTENCIA: Resultado orientativo.")
    print("  No reemplaza el criterio medico especializado.")
    print("=" * 56)
    print()


# ============================================================
# Entry point
# ============================================================

def resolve_paths(args):
    """Resuelve las rutas de los archivos .mgz segun los argumentos."""
    if args.subject_dir:
        mri_dir = os.path.join(args.subject_dir, "mri")
        aseg_path = os.path.join(mri_dir, "aparc.DKTatlas+aseg.deep.mgz")
        orig_path = os.path.join(mri_dir, "orig.mgz")
    else:
        aseg_path = args.aseg
        if not args.orig:
            print("ERROR: --orig es requerido cuando se usa --aseg")
            sys.exit(1)
        orig_path = args.orig

    for path, name in [(aseg_path, "aseg"), (orig_path, "orig")]:
        if not os.path.exists(path):
            print(f"\nERROR: No se encontro el archivo ({name}): {path}")
            sys.exit(1)

    return aseg_path, orig_path


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostico de Alzheimer para un paciente nuevo",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Entrada de imagen
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--subject_dir",
        help="Carpeta raiz del sujeto procesado por FastSurfer\n"
             "(debe contener mri/orig.mgz y mri/aparc.DKTatlas+aseg.deep.mgz)"
    )
    input_group.add_argument("--aseg", help="Ruta directa al aparc.DKTatlas+aseg.deep.mgz")
    parser.add_argument("--orig", help="Ruta al orig.mgz (requerido con --aseg)")

    # Modo de prediccion
    parser.add_argument(
        "--mode", choices=["binary", "ovr"], default="binary",
        help="Modo de prediccion:\n"
             "  binary : un solo modelo binario (default)\n"
             "  ovr    : tres modelos One-vs-Rest para diagnostico AD/MCI/CN"
    )

    # Modelos - modo binary
    parser.add_argument(
        "--model", default=None,
        help="Ruta al modelo .h5 (modo binary). Si se omite, se usa el\n"
             "primer .h5 encontrado en la carpeta inference/"
    )
    parser.add_argument(
        "--classes", default="AD,MCI",
        help="Nombres de clases para modo binary (default: AD,MCI)"
    )

    # Modelos - modo ovr
    parser.add_argument("--model_ad",  help="Modelo AD_vs_rest.h5  (modo ovr)")
    parser.add_argument("--model_mci", help="Modelo MCI_vs_rest.h5 (modo ovr)")
    parser.add_argument("--model_cn",  help="Modelo CN_vs_rest.h5  (modo ovr)")

    # Opciones comunes
    parser.add_argument(
        "--target_shape", default="100,100,100",
        help="Forma del volumen, separada por comas (default: 100,100,100)"
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Genera mapa Grad-CAM (solo disponible en modo binary)"
    )
    parser.add_argument("--save_gradcam", default=None,
                        help="Ruta para guardar la figura Grad-CAM")

    args = parser.parse_args()

    # Parsear target_shape
    try:
        target_shape = tuple(int(x) for x in args.target_shape.split(","))
        assert len(target_shape) == 3
    except Exception:
        print("ERROR: --target_shape debe tener 3 valores, ej: 100,100,100")
        sys.exit(1)

    aseg_path, orig_path = resolve_paths(args)

    print("\nArchivos de entrada:")
    print(f"  ASEG: {aseg_path}")
    print(f"  ORIG: {orig_path}")
    print(f"  Modo: {args.mode}")

    if args.mode == "ovr":
        # Verificar que se pasaron los 3 modelos
        for flag, val in [("--model_ad", args.model_ad),
                          ("--model_mci", args.model_mci),
                          ("--model_cn", args.model_cn)]:
            if not val:
                print(f"\nERROR: En modo ovr debes especificar {flag}")
                sys.exit(1)

        result = predict_ovr(
            aseg_path, orig_path,
            args.model_ad, args.model_mci, args.model_cn,
            target_shape,
        )
    else:
        class_names = [c.strip() for c in args.classes.split(",")]
        result = predict_binary(aseg_path, orig_path, args.model, target_shape, class_names)

    print_result(result)

    # Grad-CAM opcional (solo modo binary)
    if args.gradcam:
        if args.mode == "ovr":
            print("NOTA: Grad-CAM en modo OvR muestra el mapa del modelo AD.")
            model_path = args.model_ad
        else:
            model_path = args.model or _find_default_model()

        print("Generando Grad-CAM...")
        from explainability.gradcam import run_gradcam_on_subject
        run_gradcam_on_subject(
            model_path, aseg_path, orig_path,
            class_names=result["class_names"],
            save_path=args.save_gradcam,
        )


if __name__ == "__main__":
    main()
