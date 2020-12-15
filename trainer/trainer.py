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
from models.single_stream import single_stream, single_blockwise_stream
from models.double_stream import double_stream
from utils.parser import parse_args

def train(args):
    return 0