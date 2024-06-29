'''
Python script to define supporting scripts:
    1. load_json_file() : Function to load json file
    2. pixmap_to_numpy() : Function to convert pixmap to numpy array
    3. cvt_to_link() : Function to convert standard link to PySide6 format link
    4. resource_path() : Function to convert standard path to resource path necesssary while converting application to .exe using PyInstaller
    5. model_path() : Function to return the model path using the model name
    6. zoom() : Function to perform zoom
    7. zoom_in() : Function to zoom-in on the image
    8. zoom_out() : Function to zoom-out on the image
'''
import numpy as np
import os
import sys
import json
import math
import skimage.measure

def load_json_file(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


def pixmap_to_numpy(pixmap):
    ## Get the size of the current pixmap
    size = pixmap.size()
    h = size.height()
    w = size.width()
    # Get the QImage Item and convert it to a byte string
    qimg = pixmap.toImage()
    byte_str = qimg.bits().tobytes()
    # Using the np.frombuffer function to convert the byte string into an np array
    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((h, w, 4))
    return img


def cvt_to_link(text,type):
    if type==0:
        return '<a href="' + text+'">GitHub Repository</a>'
    else:
        return '<a href="' + text+'">Research Paper</a>'
    

def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def model_path(name):
    return resource_path('ONNX_models\\new_' + name.lower() + '.onnx')


def zoom_in(img, scaling_factor):
    # Repeat and reshape the image using NumPy operations
    repeated_img = np.repeat(img, scaling_factor, axis = 0)
    repeated_img = np.repeat(repeated_img, scaling_factor, axis = 1)
    return repeated_img
 
 
def zoom_out(img, scaling_factor):
    zoomed_out_img = skimage.measure.block_reduce(img, (scaling_factor, scaling_factor, 1), np.mean)
    return zoomed_out_img


def zoom(img, scaling_factor):
    loop_count = math.log(scaling_factor, 2)
    if loop_count > 0:
        img = zoom_in(img[:, :, :3], int(scaling_factor))
    else:
        img = zoom_out(img[:, :, :3], int(2 ** (-1 * loop_count)))
    return img