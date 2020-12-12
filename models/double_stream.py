# --------------------------------------------------------
# DA-RNN
# Double Stream Network
# --------------------------------------------------------
from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation, Conv2D, SeparableConv2D, MaxPooling2D, Conv2DTranspose, Softmax, Concatenate
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.nn import softmax
import numpy as np
from models.single_stream import block1, block2


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
    input1 = Input(shape=(1, ))
    input2 = Input(shape=(1, ))
    x1 = preprocessing.Resizing(128, 128, interpolation="bilinear", name="preprocess_resize")(input1)
    x1 = preprocessing.Normalization(name="preprocess_normalization")(x1)
    x2 = preprocessing.Resizing(128, 128, interpolation="bilinear", name="preprocess_resize")(input2)
    x2 = preprocessing.Normalization(name="preprocess_normalization")(x2)


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
    # block6
    # --------------------#
    split1 = Concatenate([stcut_b4_ot1, stcut_b4_ot1], axis=1, name="Concatenation_layer1")
    split2 = Concatenate([stcut_b5_2, stcut_b5_2], axis=1, name="Concatenation_layer2")


    split1 = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='_block_5_conv4')(split1)
    split1 = BatchNormalization(name='feature_extraction_block_5_bn4')(split1)


    # ----------------------------------------deconv 4 ----------------------------------------#
    split1 = Conv2DTranspose(64, (3, 3), strides=2, padding='same', kernel_initializer='glorot_uniform', activation='relu', name='embedding_deconv1', dilation_rate=(1, 1))(split1)
    split1 = BatchNormalization(name='embedding_bn1')(split1)

    split2 = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, activation='relu', padding="same", name='_block_5_conv4')(split2)
    split2 = BatchNormalization(name='feature_extraction_block_5_bn4')(split2)


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

    model = Model(inputs=input, outputs=output, name='double_stream_DA-RNN')

    return model