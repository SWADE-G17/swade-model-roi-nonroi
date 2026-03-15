"""
model/resnet3d_cbam.py

Arquitectura 3D ResNet con modulo de atencion CBAM (Convolutional Block Attention Module).
Extraida del notebook original del paper:
"Enhanced ROI guided deep learning model for Alzheimer's detection using 3D MRI images"

La arquitectura combina:
- Bloques residuales 3D (ResNet) para aprender representaciones profundas
- CBAM: atencion de canal + atencion espacial para enfocarse en las ROIs relevantes
- Kernels 5x5x5 (mas grandes que ResNet estandar) para capturar contexto espacial 3D

Uso:
    from model.resnet3d_cbam import Resnet3DBuilder

    model = Resnet3DBuilder.build_resnet_152(
        input_shape=(100, 100, 100, 1),
        num_outputs=2
    )
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import six
from math import ceil

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Activation, Dense, Flatten,
    Conv3D, MaxPooling3D, AveragePooling3D,
    BatchNormalization, GlobalAveragePooling3D,
    GlobalMaxPooling3D, Reshape
)
from tensorflow.keras.layers import add, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# Indices de los ejes (se configuran segun el backend de Keras)
DIM1_AXIS = None
DIM2_AXIS = None
DIM3_AXIS = None
CHANNEL_AXIS = None


def _handle_data_format():
    """Configura los indices de ejes segun el formato de datos de Keras."""
    global DIM1_AXIS, DIM2_AXIS, DIM3_AXIS, CHANNEL_AXIS
    if K.image_data_format() == "channels_last":
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _bn_relu(input):
    """Bloque BatchNormalization seguido de ReLU."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(**conv_params):
    """Bloque Conv3D -> BatchNorm -> ReLU."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        conv = Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
        )(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv3d(**conv_params):
    """Bloque BatchNorm -> ReLU -> Conv3D."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
        )(activation)

    return f


def _shortcut3d(input, residual):
    """
    Conexion residual (skip connection).
    Si las dimensiones no coinciden, aplica una convolucion 1x1x1 para ajustarlas.
    """
    stride_dim1 = ceil(input.shape[DIM1_AXIS] / residual.shape[DIM1_AXIS])
    stride_dim2 = ceil(input.shape[DIM2_AXIS] / residual.shape[DIM2_AXIS])
    stride_dim3 = ceil(input.shape[DIM3_AXIS] / residual.shape[DIM3_AXIS])
    equal_channels = residual.shape[CHANNEL_AXIS] == input.shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Conv3D(
            filters=residual.shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal",
            padding="valid",
            kernel_regularizer=l2(1e-4),
        )(input)
    return add([shortcut, residual])


def _residual_block_with_cbam(filters, kernel_regularizer, is_first_layer=False):
    """
    Bloque residual con modulo de atencion CBAM integrado.

    CBAM tiene dos componentes:
    1. Atencion de canal: aprende que canales (filtros) son mas importantes
    2. Atencion espacial: aprende que regiones del volumen son mas importantes

    El paper reporta que esto ayuda al modelo a enfocarse en las ROIs
    (hipocampo, sustancia blanca) y suprimir el ruido de fondo.

    Args:
        filters: numero de filtros
        kernel_regularizer: regularizacion L2
        is_first_layer: si True, usa stride (1,1,1) en vez de (2,2,2)
    """
    def f(input):
        strides = (1, 1, 1) if is_first_layer else (2, 2, 2)

        conv1 = _conv_bn_relu3D(
            filters=filters,
            kernel_size=(5, 5, 5),
            strides=strides,
            kernel_regularizer=kernel_regularizer,
        )(input)

        conv2 = _conv_bn_relu3D(
            filters=filters,
            kernel_size=(5, 5, 5),
            kernel_regularizer=kernel_regularizer,
        )(conv1)

        # --- Atencion de canal ---
        # Compara el promedio y el maximo global para decidir que canales importan
        channel_avg = GlobalAveragePooling3D()(conv2)
        channel_max = GlobalMaxPooling3D()(conv2)
        channel_attention = add([
            Dense(filters // 2, activation="relu")(channel_avg),
            Dense(filters // 2, activation="relu")(channel_max),
        ])
        channel_attention = Activation("sigmoid")(
            Dense(filters, activation="relu")(channel_attention)
        )
        channel_attention = Reshape((1, 1, 1, filters))(channel_attention)
        channel_out = multiply([conv2, channel_attention])

        # --- Atencion espacial ---
        # Aprende que voxels del volumen son mas relevantes
        spatial_attention = Conv3D(
            1, (1, 1, 1), activation="sigmoid", padding="same",
            kernel_initializer="he_normal"
        )(conv2)
        spatial_out = multiply([conv2, spatial_attention])

        # Combinar ambas atenciones
        combined = add([channel_out, spatial_out])

        return _shortcut3d(input, combined)

    return f


def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False):
    """Bloque residual basico sin CBAM (usado internamente)."""
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv3D(
                filters=filters,
                kernel_size=(5, 5, 5),
                strides=strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=kernel_regularizer,
            )(input)
        else:
            conv1 = _bn_relu_conv3d(
                filters=filters,
                kernel_size=(5, 5, 5),
                strides=strides,
                kernel_regularizer=kernel_regularizer,
            )(input)

        residual = _bn_relu_conv3d(
            filters=filters,
            kernel_size=(5, 5, 5),
            kernel_regularizer=kernel_regularizer,
        )(conv1)
        return _shortcut3d(input, residual)

    return f


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError(f"Bloque invalido: {identifier}")
        return res
    return identifier


class Resnet3DBuilder:
    """
    Constructor del modelo 3D ResNet con CBAM.

    El paper usa:
    - ROIs: build_resnet_152 con repetitions=[3, 8, 36, 6] (mas ligero)
    - Full brain: repetitions=[3, 16, 36, 6] (mas pesado)
    """

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, reg_factor=1e-4):
        """
        Construye el modelo ResNet3D+CBAM.

        Args:
            input_shape: (D, H, W, channels) - ej: (100, 100, 100, 1)
            num_outputs: numero de clases de salida
            block_fn: funcion del bloque residual
            repetitions: lista con el numero de repeticiones por etapa
            reg_factor: factor de regularizacion L2

        Returns:
            modelo Keras compilable
        """
        _handle_data_format()

        if len(input_shape) != 4:
            raise ValueError(
                "input_shape debe ser (D, H, W, channels) para channels_last"
            )

        block_fn = _get_block(block_fn)
        inputs = Input(shape=input_shape)

        # Primera convolucion
        conv1 = _conv_bn_relu3D(
            filters=128,
            kernel_size=(5, 5, 5),
            strides=(2, 2, 2),
            kernel_regularizer=l2(reg_factor),
        )(inputs)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(conv1)

        # Bloques residuales con CBAM
        block = pool1
        filters = 128
        for i, r in enumerate(repetitions):
            block = _residual_block_with_cbam(
                filters=filters,
                kernel_regularizer=l2(reg_factor),
                is_first_layer=(i == 0),
            )(block)
            filters *= 2

        block_output = _bn_relu(block)

        # Pooling global y clasificacion
        pool2 = AveragePooling3D(
            pool_size=(
                block.shape[DIM1_AXIS],
                block.shape[DIM2_AXIS],
                block.shape[DIM3_AXIS],
            ),
            strides=(1, 1, 1),
        )(block_output)
        flatten = Flatten()(pool2)

        if num_outputs > 1:
            outputs = Dense(
                units=num_outputs,
                kernel_initializer="he_normal",
                activation="softmax",
                kernel_regularizer=l2(reg_factor),
            )(flatten)
        else:
            outputs = Dense(
                units=num_outputs,
                kernel_initializer="he_normal",
                activation="sigmoid",
                kernel_regularizer=l2(reg_factor),
            )(flatten)

        return Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, reg_factor=1e-4):
        """
        ResNet152 con CBAM. Usado para ROIs en el paper.
        repetitions=[3, 8, 36, 6]
        """
        return Resnet3DBuilder.build(
            input_shape, num_outputs, basic_block, [3, 8, 36, 6], reg_factor
        )

    @staticmethod
    def build_resnet_152_full_brain(input_shape, num_outputs, reg_factor=1e-4):
        """
        ResNet152 mas pesado para full brain (mas filtros en etapas intermedias).
        repetitions=[3, 16, 36, 6]
        """
        return Resnet3DBuilder.build(
            input_shape, num_outputs, basic_block, [3, 16, 36, 6], reg_factor
        )

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, reg_factor=1e-4):
        """ResNet18 liviano para pruebas rapidas."""
        return Resnet3DBuilder.build(
            input_shape, num_outputs, basic_block, [2, 2, 2, 2], reg_factor
        )

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, reg_factor=1e-4):
        """ResNet34."""
        return Resnet3DBuilder.build(
            input_shape, num_outputs, basic_block, [3, 4, 6, 3], reg_factor
        )

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, reg_factor=1e-4):
        """ResNet101."""
        return Resnet3DBuilder.build(
            input_shape, num_outputs, basic_block, [3, 4, 23, 3], reg_factor
        )
