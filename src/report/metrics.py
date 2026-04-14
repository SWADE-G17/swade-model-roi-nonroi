"""
report/metrics.py

Funciones para evaluar el modelo entrenado y generar reportes visuales.
Incluye las mismas graficas y metricas que el notebook original del paper.

Genera:
- Curvas de accuracy y loss durante el entrenamiento
- Matriz de confusion
- Reporte de clasificacion (precision, recall, F1, AUC)

USO:
    python report/metrics.py \
        --model model_ADNI_AD_vs_MCI.h5 \
        --data  C:/alzheimer_data/processed \
        --task  AD_vs_MCI

    O desde Python:
        from report.metrics import evaluate_model, plot_training_history
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score
)

from inference.model_loading import load_model_for_inference


CLASS_NAMES = {
    "AD_vs_MCI": ["AD", "MCI"],
    "AD_vs_CN":  ["AD", "CN"],
}


def plot_training_history(history, task="AD_vs_MCI", save_path=None):
    """
    Grafica las curvas de accuracy y loss del entrenamiento.
    Identica a la celda de graficas del notebook original.

    Args:
        history: objeto History retornado por model.fit()
        task: nombre de la tarea para el titulo
        save_path: si se especifica, guarda la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title(f"Accuracy - {task}")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title(f"Loss - {task}")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Curvas de Entrenamiento - {task}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {save_path}")

    plt.show()
    return fig


def plot_confusion_matrix(true_labels, predicted_labels, class_names,
                           task="AD_vs_MCI", save_path=None):
    """
    Genera y grafica la matriz de confusion.
    Identica a la celda de confusion matrix del notebook original.

    Args:
        true_labels: array de etiquetas reales
        predicted_labels: array de etiquetas predichas
        class_names: lista con los nombres de las clases
        task: nombre de la tarea para el titulo
        save_path: si se especifica, guarda la figura
    """
    cm = confusion_matrix(true_labels, predicted_labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="g", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Etiqueta Predicha", fontsize=12)
    ax.set_ylabel("Etiqueta Real", fontsize=12)
    ax.set_title(f"Matriz de Confusion - {task}", fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Matriz guardada en: {save_path}")

    plt.show()
    return fig, cm


def print_classification_report(true_labels, predicted_labels,
                                  probabilities, class_names):
    """
    Imprime las metricas de clasificacion completas.

    Args:
        true_labels: array de etiquetas reales
        predicted_labels: array de etiquetas predichas
        probabilities: array (N, num_classes) con probabilidades
        class_names: lista con los nombres de las clases
    """
    print("\n" + "=" * 55)
    print("  REPORTE DE CLASIFICACION")
    print("=" * 55)

    print(classification_report(
        true_labels, predicted_labels,
        target_names=class_names,
        digits=4,
    ))

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy global: {accuracy:.4f} ({accuracy*100:.2f}%)")

    try:
        if probabilities.shape[1] == 2:
            auc = roc_auc_score(true_labels, probabilities[:, 1])
        else:
            auc = roc_auc_score(true_labels, probabilities, multi_class="ovr")
        print(f"AUC:             {auc:.4f}")
    except Exception:
        pass

    print("=" * 55)


def evaluate_model(model_path, base_dir, task="AD_vs_MCI",
                   target_shape=(100, 100, 100), batch_size=10,
                   save_dir=None):
    """
    Carga el modelo entrenado, evalua en el conjunto de test y genera los reportes.

    Args:
        model_path: ruta al archivo .h5 del modelo
        base_dir: ruta a los datos procesados por FastSurfer
        task: "AD_vs_MCI" o "AD_vs_CN"
        target_shape: tamano del volumen (debe coincidir con el entrenamiento)
        batch_size: tamano del batch para la prediccion
        save_dir: si se especifica, guarda las figuras en esa carpeta
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from reading.data_loader import data_generator, balance_binary_dataset

    TASK_CONFIG = {
        "AD_vs_MCI": {"classes": ["AD", "MCI"], "label_map": {"AD": 0, "MCI": 1}},
        "AD_vs_CN":  {"classes": ["AD", "CN"],  "label_map": {"AD": 0, "CN": 1}},
    }

    if task not in TASK_CONFIG:
        raise ValueError(f"task debe ser uno de {list(TASK_CONFIG.keys())}")

    config = TASK_CONFIG[task]
    classes = config["classes"]
    label_map = config["label_map"]
    class_names = CLASS_NAMES[task]

    print(f"\nCargando modelo: {model_path}")
    model = load_model_for_inference(model_path, target_shape=target_shape, compile=False)

    # Cargar datos
    paths_by_class = {}
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
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

    image_paths, labels = balance_binary_dataset(
        paths_by_class[classes[0]], paths_by_class[classes[1]],
        label_a=label_map[classes[0]], label_b=label_map[classes[1]],
    )

    _, test_paths, _, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    test_paths = np.array(test_paths)
    test_labels = np.array(test_labels)
    test_paths, test_labels = shuffle(test_paths, test_labels, random_state=42)

    print(f"Evaluando en {len(test_paths)} imagenes de test...")
    test_gen = data_generator(test_paths, test_labels, batch_size, target_shape)

    steps = len(test_paths) // batch_size
    predictions = model.predict(test_gen, steps=steps, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_labels[:len(predicted_labels)]

    print_classification_report(true_labels, predicted_labels, predictions, class_names)

    cm_path = os.path.join(save_dir, f"confusion_matrix_{task}.png") if save_dir else None
    plot_confusion_matrix(true_labels, predicted_labels, class_names, task, cm_path)

    return true_labels, predicted_labels, predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluacion del modelo Alzheimer")
    parser.add_argument("--model", required=True, help="Ruta al modelo .h5")
    parser.add_argument("--data",  required=True, help="Ruta a los datos procesados")
    parser.add_argument("--task",  default="AD_vs_MCI",
                        choices=["AD_vs_MCI", "AD_vs_CN"],
                        help="Tipo de clasificacion")
    parser.add_argument("--save_dir", default=None,
                        help="Carpeta donde guardar las figuras")
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    evaluate_model(args.model, args.data, args.task, save_dir=args.save_dir)
