""" https://www.tensorflow.org/tutorials/images/segmentation"""
"""run this file using please --image_path "data"  --dataloader "monoculer" --config_yaml "config_yaml.yml"""
import os
from pathlib import Path, PurePath, PurePosixPath
import tensorflow as tf

from IPython.display import clear_output
import matplotlib.pyplot as plt

from my_models.single_stream import single_stream, single_blockwise_stream
from my_models.double_stream import double_stream
from utils.parser import parse_args
from utils.utils import assign_dict, get_exr_depth
import yaml
import OpenEXR
import cv2
import logging
from data_generator.depth_directory_iterator import Depth_DirectoryIterator
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 选择哪一块gpu
config = ConfigProto()
config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True  # 按需分配显存，这个比较重要
session = InteractiveSession(config=config)

# pip install -q git+https://github.com/tensorflow/examples.git
# pip install -q -U tfds-nightly

logger = logging.getLogger(f"Log for {__file__}")
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
fh = logging.FileHandler(f"/{__file__}.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)


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
  y_ = model([x, x], training=training)
  y/=y.max()
  return tf.keras.losses.MSE(y_true=y, y_pred=y_)


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


if __name__ == '__main__':

    parent_path = "/home/adashao/Documents/DL-Robotics-Depth-Estimation"
    # import parser
    cfg = {}
    args = parse_args()

    f = open(parent_path + '/config_yaml.yml', 'r')
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

    for ind, val in assign_dict(cfg):
        globals()[ind] = val

    directory = Path(parent_path).joinpath(cfg["Dataset"]["DATA_PATH"])
    # enable data_generator

    #####################################  testing ##############################################
    image_data_generator = ImageDataGenerator(
        featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False, samplewise_std_normalization=False,
        zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
        height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
        channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
        horizontal_flip=False, vertical_flip=False, rescale=None,
        preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None
    )
    DirectoryIterator = Depth_DirectoryIterator(
        directory, target_size=(720, 1280),
        color_mode='rgb', output_mode='original', x_extension='png', split=0.25,
        y_extension='exr', batch_size=32, shuffle=True, seed=None, data_format=None, save_to_dir=None,
        save_prefix='', save_format='png', follow_links=False,
        interpolation='nearest', dtype=None
    )
    #
    # Generator = tf.keras.preprocessing.image.ImageDataGenerator(
    #     featurewise_center=False, samplewise_center=False,
    #     featurewise_std_normalization=False, samplewise_std_normalization=False,
    #     zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
    #     height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
    #     channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    #     horizontal_flip=False, vertical_flip=False, rescale=None,
    #     preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None
    # )
    #
    # DirectoryIterator = tf.keras.preprocessing.image.DirectoryIterator(
    #     directory, Generator, target_size=(256, 256),
    #     color_mode='rgb', classes=None, class_mode='categorical',
    #     batch_size=32, shuffle=True, seed=None, data_format=None, save_to_dir=None,
    #     save_prefix='', save_format='png', follow_links=False,
    #     subset=None, interpolation='nearest', dtype=None
    # )

    # exr_path = '/home/adashao/Documents/DL-Robotics-Praktikum/data/Seq000/depth/im0000depthgt.exr'
    # depth = get_exr_depth(Path(exr_path))
    for i, (x, y) in enumerate(DirectoryIterator):
        print(i)
        plt.figure()
        plt.subplot(121)
        plt.imshow(x[0]/x[0].max())
        plt.subplot(122)
        plt.imshow(y[0])
        plt.show()

    for x, y in DirectoryIterator:
        print(x.shape)
        print(y.shape)

    # dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


    train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset['test'].map(load_image_test)


    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    model = double_stream([128, 128, 3], 1)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    train_loss_results = []
    train_accuracy_results = []
    val_loss_results = []
    val_accuracy_results = []

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

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
