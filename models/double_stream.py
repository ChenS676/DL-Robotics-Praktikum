# --------------------------------------------------------
# DA-RNN
# Double Stream Network
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation, Conv2D, SeparableConv2D, MaxPooling2D, Conv2DTranspose, Softmax, Concatenate
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.nn import softmax
import numpy as np
from models.single_stream import block1, block2
import abc
import tensorflow as tf
import numpy as np
from tf_agents.environments import random_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils

tf.compat.v1.enable_v2_behavior()

def create_subnet(input_shape: np.ndarray, output_classes:int):
    """create downstream network in double stream network"""
    input2 = Input(shape=(input_shape))
    x2 = preprocessing.Resizing(128, 128, interpolation="bilinear", name="preprocess_resize")(input2)
    x2 = preprocessing.Normalization(name="preprocess_normalization")(x2)
    x2_block1 = block1(x2, 64, 3, 1, 2, 2, name="x2_block1")
    x2_block2 = block1(x2_block1, 128, 3, 1, 2, 2, name="x2_block2")
    x2_block3 = block2(x2_block2, 256, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=False, name="x2_block3")
    x2_block4, stcut_b4_2 = block2(x2_block3, 512, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=True, name="x2_block4")
    _, stcut_b5_2 = block2(x2_block4, 512, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=True, name="x2_block5")
    model = Model(input2, [stcut_b4_2, stcut_b5_2])
    return model


def double_stream(input_shape: np.ndarray, output_classes:int):
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
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    x1 = preprocessing.Resizing(128, 128, interpolation="bilinear", name="preprocess_resize1")(input1)
    x1 = preprocessing.Normalization(name="preprocess_normalization1")(x1)
    x2 = preprocessing.Resizing(128, 128, interpolation="bilinear", name="preprocess_resize2")(input2)
    x2 = preprocessing.Normalization(name="preprocess_normalization2")(x2)


    #--------------------#
    # block1
    #--------------------#
    x1_block1 = block1(x1, 64, 3, 1, 2, 2, name="x1_block1")
    x2_block1 = block1(x2, 64, 3, 1, 2, 2, name="x2_block1")


    #--------------------#
    # block2
    #--------------------#
    x1_block2 = block1(x1_block1, 64, 3, 1, 2, 2, name="x1_block2")
    x2_block2 = block1(x2_block1, 64, 3, 1, 2, 2, name="x2_block2")

    #--------------------#
    # block3
    #--------------------#
    x1_block3 = block2(x1_block2, 256, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=False, name="x1_block3")
    x2_block3 = block2(x2_block2, 256, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=False, name="x2_block3")


    #--------------------#
    # block4
    #--------------------#
    x1_block4, stcut_b4_ot1 = block2(x1_block3, 512, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=True, name="x1_block4")
    x2_block4, stcut_b4_ot2 = block2(x2_block3, 512, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=True, name="x2_block4")


    # --------------------#
    # block5
    # --------------------#
    _, stcut_b5_1 = block2(x1_block4, 512, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=True, name="x1_block5")
    _, stcut_b5_2 = block2(x2_block4, 512, kernal_size=3, stride=1, pooling_size=2, pooling_stride=2, shortcut=True, name="x2_block5")


    # --------------------#
    # embedding
    # --------------------#
    split1 = Concatenate()([stcut_b4_ot1, stcut_b4_ot1])
    split2 = Concatenate()([stcut_b5_2, stcut_b5_2])


    split2 = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='embedding2_conv1')(split2)
    split2 = BatchNormalization(name='embedding2_conv1_bn1')(split2)


    # ----------------------------------------deconv 4 ----------------------------------------#
    split2 = Conv2DTranspose(64, (3, 3), strides=2, padding='same', kernel_initializer='glorot_uniform', activation='relu', name='embedding_deconv1', dilation_rate=(1, 1))(split2)
    split2 = BatchNormalization(name='embedding_deconv_bn2')(split2)


    split1 = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='embedding1_v1')(split1)
    split1 = BatchNormalization(name='embedding_conv1_bn1')(split1)


    # --------------------#
    # first merge
    # --------------------#
    x = layers.add([split1, split2])

    # ----------------------------------------deconv 5  ----------------------------------------#
    x = Conv2DTranspose(64, (3, 3), strides=8, padding='same', kernel_initializer='glorot_uniform', activation='relu', name='embedding_deconv2', dilation_rate=1)(x)
    x = BatchNormalization(name='embedding_bn2')(x)


    x = Conv2D(1, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='classification_layer')(x)
    x = BatchNormalization()(x)

    output = softmax(x)

    model = Model([input1, input2], output, name='double_stream_DA-RNN')

    return model

# https://www.tensorflow.org/guide/keras/custom_layers_and_models
class Double_Stream(tf.keras.Model):
    def __init__(self,
                 preprocessing_layer=None,
                 dropout_layer_params=None,
                 name="Double_Stream"):
        super(Double_Stream, self).__init__(
            name=name
        )