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
#
from tensorflow.keras.preprocessing.image import DirectoryIterator

def _recursive_list(subpath):
    return sorted(os.walk(subpath, followlinks=False),
                  key=lambda x: x[0])


class Depth_DirectoryIterator():

    def __init__(self,
                 directory,
                 # image_data_generator,
                 target_size=(256, 256),
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
                 x_extension='exr',
                 y_extension='png',
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
        self.batch_size = batch_size
        self.seed = seed
        self.color_mode = color_mode
        self.target_size = target_size
        self.data_format = data_format
        self.batch_index = 0

        # First, count the number of samples and classes.
        self.filenames = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                self.filenames.append(subdir)
        self.num_scene = len(self.filenames)
        self.scene_indices = dict(zip(self.filenames, range(len(self.filenames))))

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        #################################################################################
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

        self.x_filepath = []
        self.y_filepath = []
        for root, _, files in _recursive_list(self.directory):
            if _ == []:
                for fname in sorted(files):
                    if fname.lower().endswith(self.x_extension):
                        self.x_filepath.append(root + fname)
                    elif fname.lower().endswith(self.y_extension):
                        self.y_filepath.append(root + fname)
        assert self.y_filepath.__len__() == self.x_filepath.__len__()

        self.index = np.arange(len(self.y_filepath)) if not self.shuffle else np.random.permutation(np.arange(len(self.y_filepath)))
        self.max_epoch_all_samples = len(self.index) / self.batch_size
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        self.epoch_index = 0
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


    def next(self):
        """# Returns
            The next batch_x, batch_y.
        """
        index_array = self.get_index_array()
        print(index_array)
        # return self.x_filepath[index_array[0]], self.y_filepath[index_array[0]]
        return index_array, index_array

    def batch_reset(self):
        self.batch_index = 0

    def set_batch_index(self):
        #  if reach the end, reset the index and reshuffle the dataset
        if self.epoch_index >= self.max_epoch_all_samples:
            logging.warning("The whole Dataset has been totally used, \
            the new shuffle will be performed, this may cause overfitting or unbalanced data weights")
            print("The whole Dataset has been totally used, \
                        the new shuffle will be performed, this may cause overfitting or unbalanced data weights")
        self.batch_index += 1
        while self.epoch_index < self.max_epoch_all_samples:
            yield self.index[(self.batch_index-1)*self.batch_size: self.batch_index*self.batch_size]


    def get_index_array(self):
        for index_array in self.set_batch_index():
            return index_array