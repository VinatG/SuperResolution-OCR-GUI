# Python module to perform Super-Resolution on input image
from scripts.model_pipelines import liff_f, srno_f, esrgan_f
import cv2
import numpy as np
from transformers import Swin2SRImageProcessor 
from scripts.model_pipelines.femasr_predictor import predictor as femasr_predictor

def esrgan(sess, image_path):
    input = esrgan_f.esrgan_input(image_path)
    pred = esrgan_f.predict(sess, input)
    return pred

def liff(sess, image_path):
    input = liff_f.liff_input(image_path)
    i1, i2, i3, i4, i5, i6 = input
    pred = liff_f.batched_predict(sess, i1, i2, i3, i4, i5, i6)
    return pred

def srno(sess, image_path):
    input = srno_f.srno_input(image_path)
    pred = srno_f.predict(sess, input)
    return pred

def img2nmp(image_path):
    image = cv2.imread(image_path)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = image_array[..., ::-1]
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, 0)
    return image_array

def output_on_pad_image(sess, output_name, inputs, window_size):
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = inputs.shape
    if h % window_size != 0:
        mod_pad_h = window_size - h % window_size
    if w % window_size != 0:
        mod_pad_w = window_size - w % window_size
    pad_width = ((0, 0), (0, 0),(0, mod_pad_h), (0, mod_pad_w))
    # Perform 'reflect' padding
    lq = np.pad(inputs, pad_width, mode='reflect')
    lq, mod_pad_h, mod_pad_w
    out_mat = sess.run([output_name], {'input' : lq})[0]
    _, _, h, w = out_mat.shape
    out_mat = out_mat[:, :, 0 : h - mod_pad_h * 4, 0 : w - mod_pad_w * 4]
    out_mat = np.squeeze(out_mat, 0)
    out_mat = np.transpose(out_mat, (1, 2, 0))
    out_mat = out_mat * 255.0
    return out_mat

def execute_sr(current_model, sess, image_path):
    if current_model == 'srno': 
        out_mat = srno(sess, image_path)         
    elif current_model == 'liif':
        out_mat = liff(sess, image_path) 
    elif current_model == 'esrgan':
        out_mat = esrgan(sess, image_path)
    elif current_model == 'swin2sr':
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processor = Swin2SRImageProcessor() 
        pixel_values = processor(image, return_tensors = "np").pixel_values
        input = sess.get_inputs()[0].name
        inputs = np.array(pixel_values)
        output_name = sess.get_outputs()[0].name
        out_mat = sess.run([output_name], {input : inputs})[0]
        out_mat = np.squeeze(out_mat, 0)
        out_mat = np.transpose(out_mat, (1, 2, 0))
        out_mat = out_mat * 255.0
        
    elif current_model.lower() in ['mdbn','pan','han','drln','a2n','dan']:
        input = sess.get_inputs()[0].name
        inputs = img2nmp(image_path)
        output_name = sess.get_outputs()[0].name
        out_mat = sess.run([output_name], {input : inputs})[0]
        out_mat = np.squeeze(out_mat, 0)
        out_mat = np.transpose(out_mat, (1, 2, 0))
        out_mat = out_mat * 255.0

    elif current_model.lower() == 'FEMASR'.lower():
        input = sess.get_inputs()[0].name
        inputs = img2nmp(image_path)
        H, W = inputs.shape[2], inputs.shape[3]
        out_mat = femasr_predictor(sess, inputs, H, W)

    elif current_model.lower() in ['dat', 'dat_s', 'dat_2', 'dat_light', 'Real_HAT_GAN_SRx4'.lower(), 'HAT-S_SRx4'.lower(), 'HAT-L_SRx4_ImageNet-pretrain'.lower(), 'HAT_SRx4_ImageNet-pretrain'.lower(), 'HAT_SRx4'.lower()]:
        input = sess.get_inputs()[0].name
        inputs = img2nmp(image_path)
        output_name = sess.get_outputs()[0].name
        window_size = 64
        out_mat = output_on_pad_image(sess ,output_name ,inputs ,window_size)

    elif current_model.lower() in ['RealworldSR-DiffIRS2x4'.lower(), 'RealworldSR-DiffIRS2-GANx4-V2'.lower(), 'RealworldSR-DiffIRS2-GANx4'.lower()]:
        input = sess.get_inputs()[0].name
        inputs = img2nmp(image_path)
        output_name = sess.get_outputs()[0].name
        window_size = 8 
        out_mat = output_on_pad_image(sess ,output_name ,inputs ,window_size)

    out_mat = (np.rint(out_mat)).astype(int)
    out_mat = np.clip(out_mat, a_min = 0, a_max = 255)
    out_mat = out_mat.astype(np.uint8)
    return out_mat