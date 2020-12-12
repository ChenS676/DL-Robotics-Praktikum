""" https://www.tensorflow.org/tutorials/images/segmentation"""
import argparse
from datetime import date
import time
import os
import numpy as np
import pandas as pd
import time
from datetime import date
from tqdm import tqdm
from pathlib import Path, PurePath, PurePosixPath
import itertools
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


import tensorflow_datasets as tfds

from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from models.single_stream import single_stream


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 选择哪一块gpu
config = ConfigProto()
config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True  # 按需分配显存，这个比较重要
session = InteractiveSession(config=config)

# def gen():
#   for i in itertools.count(1):
#     yield (i, [1] * i)
#
# dataset = tf.data.Dataset.from_generator(
#      gen,
#      (tf.int64, tf.int64),
#      (tf.TensorShape([]), tf.TensorShape([None])))

# list(dataset.take(3).as_numpy_iterator())

# pip install -q git+https://github.com/tensorflow/examples.git
# pip install -q -U tfds-nightly

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()



def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  # y_ = model(np.expand_dims(x, axis=0), training=training)
  return tf.keras.losses.MSE(y_true=y, y_pred=y_)
  # return tf.keras.losses.MSE(y_true=y, y_pred=np.expand_dims(y_, axis=0))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# show_predictions()
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 4
BUFFER_SIZE = 20
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CHANNELS = 3

EPOCHS = 100
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)


train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

model = single_stream([128, 128, 3])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


train_loss_results = []
train_accuracy_results = []
val_loss_results = []
val_accuracy_results = []

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# loss_value, grads = grad(model, image, mask)
# optimizer.apply_gradients(zip(grads, model.trainable_variables))


for epoch in range(EPOCHS):

    train_image, train_mask = train_dataset.cache().as_numpy_iterator().next()
    val_image, val_mask = test_dataset.cache().as_numpy_iterator().next()

    epoch_loss_train = tf.keras.metrics.Mean(name="train_loss")
    epoch_acc_train = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    epoch_loss_val = tf.keras.metrics.Mean(name="val_loss")
    epoch_acc_val = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")
    # Training loop - using batches of 32
    # Optimize the model

    loss_value, grads = grad(model, train_image, train_mask)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_train.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_acc_train.update_state(train_mask, model(train_image, training=True))

    # End epoch
    train_loss_results.append(epoch_loss_train.result().numpy())
    train_accuracy_results.append(epoch_acc_train.result().numpy())

    epoch_loss_val.update_state(loss(model, val_image, val_mask, True))
    epoch_acc_val.update_state(val_mask, model(val_image, training=True))

    val_loss_results.append(epoch_loss_train.result().numpy())
    val_accuracy_results.append(epoch_acc_val.result().numpy())


    if epoch % 5 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_train.result(),
                                                                    epoch_acc_train.result()))


epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, train_loss_results, 'r', label='Training loss')
plt.plot(epochs, val_loss_results, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 2])
plt.legend()
plt.show()

show_predictions(test_dataset, 3)
#
# base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
#
# # Use the activations of these layers
# layer_names = [
#     'block_1_expand_relu',  # 64x64
#     'block_3_expand_relu',  # 32x32
#     'block_6_expand_relu',  # 16x16
#     'block_13_expand_relu',  # 8x8
#     'block_16_project',  # 4x4
# ]
#
#
# layers = [base_model.get_layer(name).output for name in layer_names]
#
#
# # Create the feature extraction model
# down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
#
#
# down_stack.trainable = False
#
#
# up_stack = [
#     pix2pix.upsample(512, 3),  # 4x4 -> 8x8
#     pix2pix.upsample(256, 3),  # 8x8 -> 16x16
#     pix2pix.upsample(128, 3),  # 16x16 -> 32x32
#     pix2pix.upsample(64, 3),  # 32x32 -> 64x64
# ]
#
#

#
#
# #
# # for image, mask in train.take(1):
# #     sample_image, sample_mask = image, mask
# # display([sample_image, sample_mask])
#
# def unet_model(output_channels):
#     inputs = tf.keras.layers.Input(shape=[128, 128, 3])
#     x = inputs
#
#     # Downsampling through the model
#     skips = down_stack(x)
#     x = skips[-1]
#     skips = reversed(skips[:-1])
#
#     # Upsampling and establishing the skip connections
#     for up, skip in zip(up_stack, skips):
#         x = up(x)
#         concat = tf.keras.layers.Concatenate()
#         x = concat([x, skip])
#
#     # This is the last layer of the model
#     last = tf.keras.layers.Conv2DTranspose(
#         output_channels, 3, strides=2,
#         padding='same')  # 64x64 -> 128x128
#
#     x = last(x)
#
#     return tf.keras.Model(inputs=inputs, outputs=x)