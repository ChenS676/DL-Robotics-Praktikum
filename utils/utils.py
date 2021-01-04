import os
from pathlib import Path, PurePath, PurePosixPath
import json
import yaml
import OpenEXR
import numpy as np
import Imath
import cv2
def assign_dict(config: dict):

    last_layer = False
    # recursive find the last layer
    for index, value in config.items():
        if isinstance(value, type(config)):
            for variable in assign_dict(value):
                yield variable
        else:
            print(f"value of {index} is {value}.")

    # find the last layer
    for index, value in config.items():
        last_layer = False
        if not isinstance(value, type(dict)):
            assert not last_layer
            last_layer = True


        # assign the variables
        if last_layer:
            yield index, value


def get_exr_rgb(path):
    """https://excamera.com/articles/26/doc/openexr.html"""
    I = OpenEXR.InputFile(path)
    dw = I.header()['displayWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    data = [np.fromstring(c, np.float32).reshape(size) for c in I.channels('RGB', Imath.PixelType(Imath.PixelType.FLOAT))]
    img = np.dstack(data)
    img = np.clip(img, 0, 1)
    # convert colour to sRGB
    img = np.where(img<=0.0031308, 12.92*img, 1.055*np.power(img, 1/2.4) - 0.055)
    return (img*255).astype(np.uint8)

def get_exr_depth(filepath: Path, target_size):
    exrfile = OpenEXR.InputFile(Path(filepath).as_posix())
    raw_bytes = exrfile.channel('Z.V', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + \
        1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + \
        1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))
    depth_map = cv2.resize(depth_map, (target_size[1], target_size[0]))
    return depth_map

