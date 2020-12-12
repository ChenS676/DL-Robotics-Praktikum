# --------------------------------------------------------
# DA-RNN
# Single Stream Network
# --------------------------------------------------------

from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation, Conv2D, SeparableConv2D, MaxPooling2D, Conv2DTranspose, Softmax
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.nn import softmax
import numpy as np
from tensorflow.python.keras import backend
# def double_stream(input_shape):
#
#     return 0

# units = 32
# timesteps = 10
# input_dim = 5
# batch_size = 16


# class CustomRNN(layers.Layer):
#     def __init__(self):
#         super(CustomRNN, self).__init__()
#         self.units = units
#         self.projection_1 = layers.Dense(units=units, activation="tanh")
#         self.projection_2 = layers.Dense(units=units, activation="tanh")
#         self.classifier = layers.Dense(1)
#
#     def call(self, inputs):
#         outputs = []
#         state = tf.zeros(shape=(inputs.shape[0], self.units))
#         for t in range(inputs.shape[1]):
#             x = inputs[:, t, :]
#             h = self.projection_1(x)
#             y = h + self.projection_2(state)
#             state = y
#             outputs.append(y)
#         features = tf.stack(outputs, axis=1)
#         return self.classifier(features)


# Note that you specify a static batch size for the inputs with the `batch_shape`
# arg, because the inner computation of `CustomRNN` requires a static batch size
# (when you create the `state` zeros tensor).
# inputs = Input(batch_shape=(batch_size, timesteps, input_dim))
# x = layers.Conv1D(32, 3)(inputs)
# outputs = CustomRNN()(x)
#
# model = Model(inputs, outputs)
#
# rnn_model = CustomRNN()
# _ = rnn_model(tf.zeros((1, 10, 5)))

def block1(x, filters, kernal_size, stride, pooling_size, pooling_stride, name="None"):

    x = Conv2D(filters, (kernal_size, kernal_size), strides=(stride, stride), use_bias=False, padding="same", activation='relu',
               name=f'{name}_conv1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)

    # --------------------------------------- conv2 ---------------------------------#
    x = Conv2D(filters, (kernal_size, kernal_size), strides=(stride, stride), use_bias=False, padding="same", activation='relu',
               name=f'{name}_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)

    # --------------------------------------- pooling2 ------------------------------------------#
    x = MaxPooling2D((pooling_size, pooling_size), strides=(pooling_stride, pooling_stride), padding='same', name=f'{name}_pooling3')(x)
    output = BatchNormalization(name=f'{name}_bn3')(x)

    return output

def block2(x, filters, kernal_size=1, stride=1, pooling_size=2, pooling_stride=2, shortcut=False, name="None"):

    x = Conv2D(filters, (kernal_size, kernal_size), strides=(stride, stride), use_bias=False, activation='relu', padding="same", name=f'{name}_conv1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)

    # --------------------------------------- conv2 ------------------------------------------#
    x = Conv2D(filters, (kernal_size, kernal_size), strides=(stride, stride), use_bias=False, activation='relu', padding="same", name=f'{name}_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)

    # --------------------------------------- conv3 ------------------------------------------#
    x = Conv2D(filters, (kernal_size, kernal_size), strides=(stride, stride), use_bias=False, activation='relu', padding="same", name=f'{name}_conv3')(x)
    shortcut_ot = BatchNormalization(name=f'{name}_bn3')(x)

    # --------------------------------------- pooling2 ------------------------------------------#
    x = MaxPooling2D((pooling_size, pooling_size), strides=(pooling_stride, pooling_stride), padding='same', name=f'{name}_pooling4')(shortcut_ot)
    output = BatchNormalization(name=f'{name}_bn4')(x)

    if shortcut is True:
        return output, shortcut_ot
    else:
        return output


def single_stream(input_shape: np.ndarray):
    """Takes a single tensor as input, such as an RGB image or a depth image.
    It consists of 16 convolutional layers, 4 max pooling layers, 2 deconvolutional layers
    and 1 addition layer, all the convolutional filters in the network are of size 3*3 and stride 1.
    The max pooling layers are of size 2 *2 and stride 2. Therefore each max pooling layers reduces the
    resolution of its output. The first deconvolutional layers doubles the resolution of its output
    while the second deconvolutional layer increases the resolution by 8 times.

    -- dense pixelwise labeling
    The output of the 4th max pooling layer is 16 times smaller than the input image. The first deconvolutional
    layer doubles the resolution of its input, while the second deconvolutional layer increases the resolution by
    8 times. As a result, the output of the network has the same resolution as the input image, i.e.,
    dense pixel-wise labeling.
    """


    #--------------------------------------- preprocess ------------------------------------------#
    input = Input(shape=input_shape)
    x = preprocessing.Resizing(128, 128, interpolation="bilinear", name="preprocess_resize")(input)
    x = preprocessing.Normalization(name="preprocess_normalization")(x)


    #--------------------#
    # block1
    #--------------------#
    # --------------------------------------- conv1 ------------------------------------------#
    x = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, padding="same", activation='relu', name='feature_extraction_block_1_conv1')(x)
    x = BatchNormalization(name='feature_extraction_block_1_bn1')(x)

    # --------------------------------------- conv2 ------------------------------------------#
    x = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, padding="same", activation='relu', name='feature_extraction_block_1_conv2')(x)
    x = BatchNormalization(name='feature_extraction_block_1_bn2')(x)

    # --------------------------------------- pooling2 ------------------------------------------#
    # (x-4)/2
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='feature_extraction_block_1_pooling3')(x)
    x_block1 = BatchNormalization(name='feature_extraction_block_1_bn_3')(x)
    # 318, 248

    #--------------------#
    # block2
    #--------------------#
    # --------------------------------------- conv1 ------------------------------------------#
    x = Conv2D(128, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_2_conv1')(x_block1)
    x = BatchNormalization(name='feature_extraction_block_2_bn1')(x)


    # --------------------------------------- conv2 ------------------------------------------#
    x = Conv2D(128, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_2_conv2')(x)
    x = BatchNormalization(name='feature_extraction_block_2_bn2')(x)

    # --------------------------------------- pooling2 ------------------------------------------#
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='feature_extraction_block_2_pooling3')(x)
    x_block2 = BatchNormalization(name='feature_extraction_block_2_bn_3')(x)


    #--------------------#
    # block3
    #--------------------#
    # --------------------------------------- conv1 ------------------------------------------#
    x = Conv2D(256, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_3_conv1')(x_block2)
    x = BatchNormalization(name='feature_extraction_block_3_bn1')(x)

    # --------------------------------------- conv2 ------------------------------------------#
    x = Conv2D(256, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_3_conv2')(x)
    x = BatchNormalization(name='feature_extraction_block_3_bn2')(x)

    # --------------------------------------- conv3 ------------------------------------------#
    x = Conv2D(256, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_3_conv3')(x)
    x = BatchNormalization(name='feature_extraction_block_3_bn3')(x)

    # --------------------------------------- pooling2 ------------------------------------------#
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='feature_extraction_block_3_pooling4')(x)
    x_block3 = BatchNormalization(name='feature_extraction_block_3_bn_4')(x)

    # --------------------#
    # block4
    # --------------------#
    # --------------------------------------- conv1 ------------------------------------------#
    x = Conv2D(512, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_4_conv1')(x_block3)
    x = BatchNormalization(name='feature_extraction_block_4_bn1')(x)


    # --------------------------------------- conv2 ------------------------------------------#
    x = Conv2D(512, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_4_conv2')(x)
    x = BatchNormalization(name='feature_extraction_block_4_bn2')(x)


    # --------------------------------------- conv3 ------------------------------------------#
    x = Conv2D(512, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_4_conv3')(x)
    residual = BatchNormalization(name='feature_extraction_block_4_bn3')(x)

    # --------------------#
    # first split
    # --------------------#
    x1 = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False,  activation='relu', padding="same", name='split_block_4_conv4')(residual)
    add2 = BatchNormalization(name='split_block_4_bn4')(x1)

    # --------------------#
    # second split
    # --------------------#
    # --------------------------------------- pooling2 ------------------------------------------#
    x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='feature_extraction_block_4_pooling4')(residual)
    x_block4 = BatchNormalization(name='feature_extraction_block_4_bn_4')(x2)

    # --------------------#
    # block5
    # --------------------#
    # --------------------------------------- conv1 ------------------------------------------#
    x2 = Conv2D(512, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_5_conv1')(x_block4)
    x2 = BatchNormalization(name='feature_extraction_block_5_bn1')(x2)

    # --------------------------------------- conv2 ------------------------------------------#
    x2 = Conv2D(512, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_5_conv2')(x2)
    x2 = BatchNormalization(name='feature_extraction_block_5_bn2')(x2)


    # --------------------------------------- conv3 ------------------------------------------#
    x2 = Conv2D(512, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_5_conv3')(x2)
    x2 = BatchNormalization(name='feature_extraction_block_5_bn3')(x2)


    x2 = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='feature_extraction_block_5_conv4')(x2)
    x2 = BatchNormalization(name='feature_extraction_block_5_bn4')(x2)

    # ----------------------------------------deconv 4 ----------------------------------------#
    x2 = Conv2DTranspose(64, (3, 3), strides=2, padding='same', kernel_initializer='glorot_uniform', activation='relu', name='embedding_deconv1', dilation_rate=(1, 1))(x2)
    add1 = BatchNormalization(name='embedding_bn1')(x2)

    # --------------------#
    # first merge
    # --------------------#
    x = layers.add([add1, add2])

    # ----------------------------------------deconv 5  ----------------------------------------#
    x = Conv2DTranspose(64, (3, 3), strides=8, padding='same', kernel_initializer='glorot_uniform', activation='relu', name='embedding_deconv2', dilation_rate=1)(x)
    x = BatchNormalization(name='embedding_bn2')(x)


    x = Conv2D(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='classification_layer')(x)
    x = BatchNormalization()(x)

    output = softmax(x)

    model = Model(inputs=input, outputs=output, name='single_stream_DA-RNN')

    # model.load_weights("xception_weights_tf_dim_ordering_tf_kernels.h5")

    return model

def single_blockwise_stream(input_shape: np.ndarray, output_classes: int):
    """Takes a single tensor as input, such as an RGB image or a depth image.
    It consists of 16 convolutional layers, 4 max pooling layers, 2 deconvolutional layers
    and 1 addition layer, all the convolutional filters in the network are of size 3*3 and stride 1.
    The max pooling layers are of size 2 *2 and stride 2. Therefore each max pooling layers reduces the
    resolution of its output. The first deconvolutional layers doubles the resolution of its output
    while the second deconvolutional layer increases the resolution by 8 times.

    -- dense pixelwise labeling
    The output of the 4th max pooling layer is 16 times smaller than the input image. The first deconvolutional
    layer doubles the resolution of its input, while the second deconvolutional layer increases the resolution by
    8 times. As a result, the output of the network has the same resolution as the input image, i.e.,
    dense pixel-wise labeling.
    """


    #--------------------------------------- preprocess ------------------------------------------#
    input = Input(shape=input_shape)
    x = preprocessing.Resizing(128, 128, interpolation="bilinear", name="preprocess_resize")(input)
    x = preprocessing.Normalization(name="preprocess_normalization")(x)


    #--------------------#
    # block1
    #--------------------#
    x_block1 = block1(x, 64, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, name="feature_extraction_block1")
    #--------------------#
    # block2
    #--------------------#
    x_block2 = block1(x_block1, 128, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, name="feature_extraction_block2")
    #--------------------#
    # block3
    #--------------------#
    x_block3 = block2(x_block2, 256, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=False, name="feature_Extraction_block3")
    #--------------------#
    # block4
    #--------------------#
    x_block4, shortcut_ot = block2(x_block3, 512, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=True, name="feature_Extraction_block4")
    # --------------------#
    # first split
    # --------------------#
    x1 = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False,  activation='relu', padding="same", name='split_block_4_conv4')(shortcut_ot)
    add2 = BatchNormalization(name='split_block_4_bn4')(x1)

    # --------------------#
    # second split
    # --------------------#
    # --------------------#
    # block5
    # --------------------#
    _, shortcut = block2(x_block4, 512, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=True, name="feature_Extraction_block5")

    # ----------------------------------------deconv 4 ----------------------------------------#
    x2 = Conv2DTranspose(64, (3, 3), strides=2, padding='same', kernel_initializer='glorot_uniform', activation='relu', name='embedding_deconv1', dilation_rate=(1, 1))(shortcut)
    add1 = BatchNormalization(name='embedding_bn1')(x2)

    # --------------------#
    # first merge
    # --------------------#
    x = layers.add([add1, add2])

    # ----------------------------------------deconv 5  ----------------------------------------#
    x = Conv2DTranspose(64, (3, 3), strides=8, padding='same', kernel_initializer='glorot_uniform', activation='relu', name='embedding_deconv2', dilation_rate=1)(x)
    x = BatchNormalization(name='embedding_bn2')(x)


    x = Conv2D(output_classes, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='classification_layer')(x)
    x = BatchNormalization()(x)

    output = softmax(x)

    model = Model(inputs=input, outputs=output, name='single_stream_DA-RNN')

    # model.load_weights("xception_weights_tf_dim_ordering_tf_kernels.h5")

    return model