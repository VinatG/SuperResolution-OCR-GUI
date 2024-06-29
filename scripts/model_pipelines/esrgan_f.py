# Python module that defines script to processs input image for the ESRGAN model and perform the Super-Resolution
import numpy as np
import cv2
import onnxruntime as rt
import numpy as np

def esrgan_input(file_path):
    in_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    in_mat = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
    in_mat = np.transpose(in_mat, (2, 1, 0))[np.newaxis]
    in_mat = in_mat.astype(np.float32)
    in_mat = in_mat / 255
    return in_mat

def predict(sess, input):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    out_mat = sess.run([output_name], {input_name : input})[0]
    out_mat = np.squeeze(out_mat, axis = 0)
    out_mat = np.clip(out_mat, 0, 1)
    out_mat = (out_mat * 255.).round().astype(np.uint8)
    out_mat = out_mat.T
    return out_mat



