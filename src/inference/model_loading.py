"""
Carga de modelos .h5 para inferencia.

Algunos checkpoints se guardaron con otra version de Keras/TensorFlow y
`load_model` falla al deserializar la arquitectura (p. ej. argumentos
`batch_shape`, `optional`). En ese caso se reconstruye ResNet152 3D + CBAM
igual que en training/train.py y se cargan solo los pesos.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model_for_inference(model_path, target_shape=(100, 100, 100), compile=False):
    """
    Carga el modelo completo o, si falla la deserializacion, arquitectura + pesos.

    Args:
        model_path: ruta al .h5
        target_shape: (D, H, W) del volumen, debe coincidir con el entrenamiento
        compile: se pasa a load_model cuando aplica (inferencia suele usar False)
    """
    import tensorflow as tf
    from model.resnet3d_cbam import Resnet3DBuilder

    try:
        return tf.keras.models.load_model(model_path, compile=compile)
    except Exception as e:
        msg = str(e).lower()
        triggers = ("batch_shape", "optional", "unrecognized keyword")
        if not any(t in msg for t in triggers):
            raise

        model = Resnet3DBuilder.build_resnet_152(
            input_shape=(*target_shape, 1),
            num_outputs=2,
        )
        try:
            model.load_weights(model_path)
        except Exception:
            model.load_weights(model_path, by_name=True)
        return model
