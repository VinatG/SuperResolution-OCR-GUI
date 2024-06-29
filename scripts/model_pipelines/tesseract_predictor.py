#Python module to run pytesseract on input image using the tesseract engine
import cv2
import numpy as np
from pytesseract import Output
import pytesseract

def predict(image):
    # Get bounding boxes of the text
    if len(image.shape) == 4:
        image = np.squeeze(image, axis=0)
        image = np.transpose(image, (1, 2, 0))
    img_h, img_w, _ = image.shape
    c_image = np.ascontiguousarray(image, dtype=np.uint8)
    d = pytesseract.image_to_data(c_image, output_type = Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 50:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            text = d['text'][i]
            base_font_scale = 2
            base_thickness = 1
            font_scale = min( w / 100,  h / 100) * base_font_scale
            thickness = round( font_scale * 1.1)
            c_image = cv2.rectangle(c_image, (x - 1, y), (x + w + 1, y + h), (0, 255, 0), 1)
            cv2.putText(c_image, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
    return c_image