# Python module that defines script to processs input image for the Liff model and perform the Super-Resolution
from time import time
import numpy as np
from PIL import Image
import numpy as np

def make_coord(shape, ranges=None, flatten=True):
    #Make coordinates at grid centers
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * np.arange(n, dtype=float)
        coord_seqs.append(seq)
    meshgrid = np.meshgrid(*coord_seqs, indexing='ij')
    ret = np.stack(meshgrid, axis=-1)
    if flatten:
        ret = np.reshape(ret, (-1, ret.shape[-1]))
    return ret

def liff_input(file_path):
    image = Image.open(file_path).convert('RGB')
    in_image = np.array(image)
    in_image = in_image / 255.0
    img = in_image.transpose(2, 0, 1)

    scale_f = 4
    scale_max = 4
    h = int(img.shape[-2] * int(scale_f))
    w = int(img.shape[-1] * int(scale_f))
    scale = h / img.shape[-2]
    coord = make_coord((h, w))
    cell = np.ones_like(coord)

    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    cell_factor = max(scale/scale_max, 1)
    input=[
    np.expand_dims(((img - 0.5) / 0.5), axis = 0),
    np.expand_dims(coord, axis = 0),
    np.expand_dims(cell_factor * cell,axis = 0),
    30000, h, w]
    input[0] = input[0].astype(np.float32)
    input[1] = input[1].astype(np.float32)
    input[2] = input[2].astype(np.float32)
    return input

def batched_predict(sess, inp, coord, cell, bsize, h, w):
    n = coord.shape[1]
    ql = 0
    preds = []
    while ql < n:
        qr = min(ql + bsize, n)
        i2 = coord[:, ql : qr, :]
        i3 = cell[:, ql : qr, :]
        input1 = sess.get_inputs()[0].name
        input2 = sess.get_inputs()[1].name
        input3 = sess.get_inputs()[2].name
        output_name = sess.get_outputs()[0].name
        inp = np.array(inp)
        i2 = np.array(i2)
        i3 = np.array(i3)
        input = (inp, i2, i3, np.array(ql), np.array(qr))
        pred = sess.run([output_name], {input1 : input[0], input2 : input[1], input3 : input[2]})[0]
        preds.append(pred)
        ql = qr

    pred = np.concatenate(preds, axis=1)
    pred = pred[0]
    pred = (pred * 0.5 + 0.5).clip(0, 1).reshape((h, w, 3)).transpose(2, 0, 1)
    pred = np.array(pred)
    pred = np.clip(pred, 0, 1)
    pred = (pred * 255.).round().astype(np.uint8)
    pred = np.transpose(pred, (1, 2, 0))
    return pred

