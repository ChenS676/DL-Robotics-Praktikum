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
import multiprocessing

try:
    IteratorType = get_keras_submodule('utils').Sequence
except ImportError:
    IteratorType = object

from keras.preprocessing.image import (array_to_img, img_to_array, load_img)
from .iterator import BatchFromFilesMixin, Iterator
from tensorflow.keras.preprocessing.image import DirectoryIterator
from .utils import _list_valid_filenames_in_directory



class Depth_DirectoryIterator():
    y_output = {'original', 'scaled', 'reshaped', None}

    def __init__(self,
                 directory,
                 image_data_generator,
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
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        # super(Iterator, self).set_processing_attrs(image_data_generator,
        #                                                     target_size,
        #                                                     color_mode,
        #                                                     data_format,
        #                                                     save_to_dir,
        #                                                     save_prefix,
        #                                                     save_format,
        #                                                     subset,
        #                                                     interpolation)

        self.directory = directory
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.dtype = dtype


        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        # super(Depth_DirectoryIterator, self).__init__(self.samples,
        #                                         batch_size,
        #                                         shuffle,
        #                                         seed)



