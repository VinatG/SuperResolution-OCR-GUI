# Python module that defines script to processs input image for the SRNO model and perform the Super-Resolution
from PIL import Image
import numpy as np

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) *np.arange(n, dtype=float)
        coord_seqs.append(seq)
    meshgrid = np.meshgrid(*coord_seqs, indexing='ij')
    ret = np.stack(meshgrid, axis=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def srno_input(file_path):
    scale_max = 4
    image = Image.open(file_path).convert('RGB')
    in_image = np.array(image)
    # Normalize pixel values to the range [0, 1]
    in_image = in_image / 255.0
    in_image = in_image.transpose(2, 0, 1)
    scale_f = 4
    h = int(in_image.shape[-2] * int(scale_f))
    w = int(in_image.shape[-1] * int(scale_f))
    scale = h / in_image.shape[-2]
    coord = make_coord((h, w), flatten=False)
    cell = np.ones((1, 2))
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    cell_factor = max(scale / scale_max, 1)
    input=[
        np.expand_dims(((in_image - 0.5) / 0.5), axis = 0),
        np.expand_dims(coord, axis = 0), 
        cell_factor * cell]
    input[0] = input[0].astype(np.float32)
    input[1] = input[1].astype(np.float32)
    input[2] = input[2].astype(np.float32)
    return input

def predict(sess, input):
    input1 = sess.get_inputs()[0].name
    input2 = sess.get_inputs()[1].name
    input3 = sess.get_inputs()[2].name
    input[1] = np.array(input[1])
    input[2] = np.array(input[2])
    output_name = sess.get_outputs()[0].name
    out_mat = sess.run([output_name], {input1 : input[0], input2 : input[1], input3 : input[2]})[0]
    pred = out_mat.squeeze(0)
    pred = (pred * 0.5 + 0.5).clip(0, 1)
    pred=pred[:, :, ::-1]
    pred = np.transpose(pred, (1, 2, 0))
    pred = pred * 255.0
    pred = np.flip(pred, axis=1)
    return pred

