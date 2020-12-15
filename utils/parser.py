from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parser for Depth Estimation Project.')

    parser.add_argument('--image_path', type=str,
                        help='path of datasets', required=True)
    # To Do
    parser.add_argument("--dataloader", type=str,
                        help='type of dataloader', required=False,
                        choices=["monoculer", "multi-view", "binocular"])

    parser.add_argument('--model_name', type=str,
                        help='name of a model to use/train',
                        choices=[
                            "monodepth2",
                            "DA-RNN_single_stream",
                            "DA-RNN_double_stream"])

    parser.add_argument('--config_yaml', type=str,
                        help="configuration file for the trainer/evaluater/predictor",
                        required=False)

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")

    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()