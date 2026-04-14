"""
training/train.py

Script de entrenamiento del modelo 3D ResNet + CBAM para deteccion de Alzheimer.
Equivale al notebook original pero organizado como script Python ejecutable.

USO (desde la raiz del proyecto):
    conda activate alzheimer
    python training/train.py

CONFIGURACION:
    Edita las variables en la seccion CONFIGURACION mas abajo.

SALIDA:
    - Archivo .h5 con el mejor modelo segun val_accuracy
    - Archivo .png con las curvas de entrenamiento
    - Archivo .png con la matriz de confusion

REQUISITOS:
    tensorflow==2.10.0, nibabel, scikit-image, opencv-python,
    dipy, scikit-learn, matplotlib, seaborn
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Agregar la raiz del proyecto al path para importar los modulos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

from model.resnet3d_cbam import Resnet3DBuilder
from reading.data_loader import load_dataset, data_generator, balance_binary_dataset, prepare_ovr_dataset

# ============================================================
# CONFIGURACION - EDITA ESTAS VARIABLES ANTES DE CORRER
# ============================================================

# Ruta a los datos procesados por FastSurfer
# Debe contener subcarpetas AD/, MCI/, CN/ con los sujetos procesados
BASE_DIR = r"C:\Users\Harry\Desktop\processed"

# Tipo de clasificacion. Opciones:
#
#   Clasificacion binaria (paper original):
#     "AD_vs_MCI"   -> distingue AD de MCI
#     "AD_vs_CN"    -> distingue AD de CN
#     "MCI_vs_CN"   -> distingue MCI de CN
#
#   One-vs-Rest (nuevo, para diagnostico de 3 clases):
#     "AD_vs_rest"  -> AD vs (MCI + CN)   <- entrenar primero
#     "MCI_vs_rest" -> MCI vs (AD + CN)   <- entrenar segundo
#     "CN_vs_rest"  -> CN vs (AD + MCI)   <- entrenar tercero
#
# Para el diagnostico final de 3 clases, entrena los 3 modelos OvR
# y luego usa inference/predict.py con --mode ovr

TASK = "AD_vs_rest"

# Nombre del archivo donde se guarda el mejor modelo
MODEL_FILENAME = "model_ADNI_" + TASK + ".h5"

# Tamano del volumen de entrada al modelo (no cambiar)
TARGET_SHAPE = (100, 100, 100)

# Numero de imagenes por batch (reducir a 4 si hay problemas de memoria GPU)
BATCH_SIZE = 1

# Numero de epocas de entrenamiento
NUM_EPOCHS = 100

# Fraccion de datos para evaluacion (0.2 = 20%)
TEST_SIZE = 0.2

# Semilla para reproducibilidad
RANDOM_SEED = 42

# ============================================================
# NO MODIFICAR LO DE ABAJO
# ============================================================

# Tareas binarias clasicas (igual que el paper)
TASK_CONFIG = {
    "AD_vs_MCI": {
        "classes": ["AD", "MCI"],
        "label_map": {"AD": 0, "MCI": 1},
        "class_names": ["AD", "MCI"],
        "mode": "binary",
    },
    "AD_vs_CN": {
        "classes": ["AD", "CN"],
        "label_map": {"AD": 0, "CN": 1},
        "class_names": ["AD", "CN"],
        "mode": "binary",
    },
    "MCI_vs_CN": {
        "classes": ["MCI", "CN"],
        "label_map": {"MCI": 0, "CN": 1},
        "class_names": ["MCI", "CN"],
        "mode": "binary",
    },
    # One-vs-Rest: etiqueta 1 = clase positiva, 0 = el resto
    "AD_vs_rest": {
        "positive_class": "AD",
        "class_names": ["rest", "AD"],
        "mode": "ovr",
    },
    "MCI_vs_rest": {
        "positive_class": "MCI",
        "class_names": ["rest", "MCI"],
        "mode": "ovr",
    },
    "CN_vs_rest": {
        "positive_class": "CN",
        "class_names": ["rest", "CN"],
        "mode": "ovr",
    },
}


def configure_gpu():
    """Configura la GPU para crecimiento dinamico de memoria."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {len(gpus)} GPU(s)")
    else:
        print("ADVERTENCIA: No se detecto GPU. El entrenamiento sera muy lento.")


def prepare_data(base_dir, task_config):
    """
    Carga, balancea y divide los datos en train/test.
    Soporta tanto tareas binarias como One-vs-Rest (OvR).

    Returns:
        train_paths, test_paths, train_labels, test_labels, class_weights
    """
    print(f"\nCargando datos desde: {base_dir}")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(
            f"No existe la carpeta: {base_dir}\n"
            "Verifica que FastSurfer haya procesado los datos y que BASE_DIR sea correcto."
        )

    mode = task_config.get("mode", "binary")

    if mode == "ovr":
        # One-vs-Rest: positivo vs (las otras dos clases juntas)
        positive_class = task_config["positive_class"]
        image_paths, labels, _ = prepare_ovr_dataset(
            base_dir, positive_class, random_seed=RANDOM_SEED
        )
    else:
        # Binaria clasica: clase A vs clase B
        classes = task_config["classes"]
        label_map = task_config["label_map"]

        paths_by_class = {}
        for class_name in classes:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"No existe la carpeta: {class_dir}")
            paths = []
            for subject_dir in sorted(os.listdir(class_dir)):
                subject_path = os.path.join(class_dir, subject_dir)
                if not os.path.isdir(subject_path):
                    continue
                mri_dir = os.path.join(subject_path, "mri")
                aseg = os.path.join(mri_dir, "aparc.DKTatlas+aseg.deep.mgz")
                orig = os.path.join(mri_dir, "orig.mgz")
                if os.path.exists(aseg) and os.path.exists(orig):
                    paths.append(aseg)
            paths_by_class[class_name] = paths
            print(f"  {class_name}: {len(paths)} sujetos")

        class_a, class_b = classes[0], classes[1]
        image_paths, labels = balance_binary_dataset(
            paths_by_class[class_a], paths_by_class[class_b],
            label_a=label_map[class_a],
            label_b=label_map[class_b],
        )

    print(f"\nTotal tras balanceo: {len(image_paths)} imagenes")

    # Split train/test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Shuffle
    train_paths = np.array(train_paths)
    train_labels = np.array(train_labels)
    test_paths = np.array(test_paths)
    test_labels = np.array(test_labels)

    train_paths, train_labels = shuffle(train_paths, train_labels, random_state=RANDOM_SEED)
    test_paths, test_labels = shuffle(test_paths, test_labels, random_state=RANDOM_SEED)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )

    print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")
    print(f"Class weights: {class_weights}")

    return train_paths, test_paths, train_labels, test_labels, class_weights


def build_model(target_shape, num_classes):
    """Construye y compila el modelo ResNet152 + CBAM."""
    input_shape = (*target_shape, 1)
    model = Resnet3DBuilder.build_resnet_152(
        input_shape=input_shape,
        num_outputs=num_classes,
    )
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss="binary_crossentropy",
        metrics=["accuracy", "Recall", "AUC", "Precision"],
    )
    return model


def train(model, train_paths, test_paths, train_labels, test_labels,
          target_shape, batch_size, num_epochs, model_filename):
    """Entrena el modelo y guarda el mejor checkpoint."""
    train_gen = data_generator(
        train_paths, train_labels, batch_size, target_shape, is_training=True
    )
    test_gen = data_generator(
        test_paths, test_labels, batch_size, target_shape, is_training=False
    )

    checkpoint = ModelCheckpoint(
        model_filename,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    print(f"\nEntrenando por {num_epochs} epocas...")
    print(f"El mejor modelo se guardara en: {model_filename}")

    history = model.fit(
        train_gen,
        epochs=num_epochs,
        steps_per_epoch=len(train_paths) // batch_size,
        validation_data=test_gen,
        validation_steps=len(test_paths) // batch_size,
        callbacks=[checkpoint],
    )

    return history


def main():
    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO: Alzheimer Detection - 3D ResNet + CBAM")
    print("=" * 60)
    print(f"  Tarea:    {TASK}")
    print(f"  Datos:    {BASE_DIR}")
    print(f"  Modelo:   {MODEL_FILENAME}")
    print(f"  Epocas:   {NUM_EPOCHS}")
    print(f"  Batch:    {BATCH_SIZE}")
    print("=" * 60)

    if TASK not in TASK_CONFIG:
        print(f"ERROR: TASK debe ser uno de {list(TASK_CONFIG.keys())}")
        sys.exit(1)

    task_config = TASK_CONFIG[TASK]

    configure_gpu()

    train_paths, test_paths, train_labels, test_labels, class_weights = prepare_data(
        BASE_DIR, task_config
    )

    print("\nConstruyendo modelo...")
    model = build_model(TARGET_SHAPE, num_classes=2)

    history = train(
        model, train_paths, test_paths, train_labels, test_labels,
        TARGET_SHAPE, BATCH_SIZE, NUM_EPOCHS, MODEL_FILENAME,
    )

    print(f"\nEntrenamiento completado. Modelo guardado en: {MODEL_FILENAME}")

    task_config = TASK_CONFIG[TASK]
    if task_config.get("mode") == "ovr":
        print()
        print("Para el diagnostico de 3 clases, entrena tambien los otros dos modelos:")
        remaining = [t for t in ["AD_vs_rest", "MCI_vs_rest", "CN_vs_rest"] if t != TASK]
        for t in remaining:
            print(f"  Cambia TASK = \"{t}\" y vuelve a correr este script")
        print()
        print("Cuando tengas los 3 modelos, usa inference/predict.py con --mode ovr")

    print("\nPara ver las metricas: python report/metrics.py --model", MODEL_FILENAME,
          "--data", BASE_DIR, "--task", TASK)


if __name__ == "__main__":
    main()
