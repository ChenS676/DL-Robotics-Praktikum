"""Utilities for real-time data augmentation on image data.
    Inherit from tensorflow.python.keras.preprocessing.image.directoryiterator

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import numpy as np
from keras_preprocessing import get_keras_submodule
import multiprocessing.pool
import warnings
try:
    IteratorType = get_keras_submodule('utils').Sequence
except ImportError:
    IteratorType = object
import logging

from pathlib import Path, PurePath, PurePosixPath
from random import shuffle
#
from keras.preprocessing.image import (array_to_img, img_to_array, load_img)
from tensorflow.python.keras import backend
from tensorflow.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.preprocessing.image import Iterator
from utils.utils import  get_exr_depth
def _recursive_list(subpath):
    return sorted(os.walk(subpath, followlinks=False),
                  key=lambda x: x[0])


class Depth_DirectoryIterator():
    def __init__(self,
                 directory,
                 # image_data_generator,
                 target_size=(128, 128),
                 color_mode='rgb',
                 output_mode='original',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 x_extension='png',
                 y_extension='exr',
                 split=None,
                 interpolation='nearest',
                 dtype='float32'):

        self.allowed_output = {'original', 'scaled', 'reshaped', None}

        self.directory = directory
        self.output_mode = output_mode
        if output_mode not in self.allowed_output:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(output_mode, self.allowed_output))
        self.output_mode = output_mode
        self.dtype = dtype

        self.x_extension = x_extension
        self.y_extension = y_extension
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.color_mode = color_mode
        self.target_size = target_size
        self.data_format = data_format
        self.batch_index = 0
        self.sample_weight = None
        self.batch_size = batch_size
        self.seed = seed
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

        # First, count the number of samples and classes.
        self.filenames = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                self.filenames.append(subdir)
        self.num_scene = len(self.filenames)
        self.scene_indices = dict(zip(self.filenames, range(len(self.filenames))))

        self.x_filepath = []
        self.y_filepath = []
        for root, _, files in _recursive_list(self.directory):
            if _ == []:
                for fname in sorted(files):
                    if fname.lower().endswith(self.x_extension):
                        self.x_filepath.append(root + '/' + fname)
                    elif fname.lower().endswith(self.y_extension):
                        self.y_filepath.append(root + '/' + fname)
        assert self.y_filepath.__len__() == self.x_filepath.__len__()

        self.index = np.arange(len(self.y_filepath)) if not self.shuffle else np.random.permutation(np.arange(len(self.y_filepath)))
        self.max_epoch_all_samples = len(self.index) / self.batch_size
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        self.n = len(self.y_filepath)
        self.interpolation = interpolation
        self.save_to_dir = save_to_dir,
        self.save_prefix = save_prefix,
        self.save_format = save_format,
        self.follow_links = follow_links,
        if data_format is None:
            data_format = backend.image_data_format()
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation


    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        """When you want to use x[index_array], y[index_array]"""
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)


    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                # Avoiding modulo by zero error
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 3
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        batch_y = np.zeros((len(index_array),) + self.target_size, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.x_filepath
        for i, j in enumerate(index_array):
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            # if self.image_data_generator:
            #     params = self.image_data_generator.get_random_transform(x.shape)
            #     x = self.image_data_generator.apply_transform(x, params)
            #     x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # build batch of y
        filepaths = self.y_filepath
        for i, j in enumerate(index_array):
            y = get_exr_depth(filepaths[j], self.target_size)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            batch_y[i] = y


        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

        # Second, build an index of the images
        # in the different class subfolders.
        #################################################################################
        # pool = multiprocessing.pool.ThreadPool()
        # classes_list = []
        # for res in results:
        #     classes, filenames = res.get()
        #     classes_list.append(classes)
        #     self.filenames += filenames
        # self.samples = len(self.filenames)
        # self.classes = np.zeros((self.samples,), dtype='int32')
        # for classes in classes_list:
        #     self.classes[i:i + len(classes)] = classes
        #     i += len(classes)
        #
        # print('Found %d images belonging to %d classes.' %
        #       (self.samples, self.num_classes))
        # pool.close()
        # pool.join()
        ####################################################################################



        # def __iter__(self):
        #     # Needed if we want to do something like:
        #     # for x, y in data_gen.flow(...):
        #     self.epoch_index = 0
        #     return self
        #
        # def __next__(self, *args, **kwargs):
        #     return self.next(*args, **kwargs)
        #
        #
        # def next(self):
        #    """# Returns
        #         The next batch_x, batch_y.
        #     """
        #     index_array = self.get_index_array()
        #     print(index_array)
        #     # return self.x_filepath[index_array[0]], self.y_filepath[index_array[0]]
        #     return index_array, index_array
        #
        # def batch_reset(self):
        #     self.batch_index = 0
        #
        # def set_batch_index(self):
        #     #  if reach the end, reset the index and reshuffle the dataset
        #     if self.epoch_index >= self.max_epoch_all_samples:
        #         logging.warning("The whole Dataset has been totally used, \
        #         the new shuffle will be performed, this may cause overfitting or unbalanced data weights")
        #         print("The whole Dataset has been totally used, \
        #                     the new shuffle will be performed, this may cause overfitting or unbalanced data weights")
        #     self.batch_index += 1
        #     while self.epoch_index < self.max_epoch_all_samples:
        #         yield self.index[(self.batch_index-1)*self.batch_size: self.batch_index*self.batch_size]
        #
        #
        # def get_index_array(self):
        #     for index_array in self.set_batch_index():
        #         return index_array

        # white_list_formats = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')