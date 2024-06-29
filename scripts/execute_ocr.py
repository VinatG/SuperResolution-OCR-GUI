# Python module to perform OCR on input image
from scripts.model_pipelines.tesseract_predictor import predict as tesseract_ocr_predict
from scripts.utils import resource_path
import pytesseract 
from PaddleOCR.tools.infer.predict_system import paddleocr_predict as paddle_ocr_predict
pytesseract.pytesseract.tesseract_cmd = resource_path(r'./Tesseract_OCR/tesseract.exe')

def execute_ocr(current_ocr_model, out_mat, inp_img_array):
    if current_ocr_model.lower() == 'pytesseract':
        sr_ocr = tesseract_ocr_predict(out_mat)
        lr_ocr = tesseract_ocr_predict(inp_img_array)

    elif current_ocr_model.lower() == 'paddleocr':
        sr_ocr = paddle_ocr_predict(out_mat, './ONNX_models/paddle_det_model.onnx', './ONNX_models/paddle_rec_model.onnx')
        lr_ocr = paddle_ocr_predict(inp_img_array, './ONNX_models/paddle_det_model.onnx', './ONNX_models/paddle_rec_model.onnx')
    else:
        lr_ocr = inp_img_array
        sr_ocr = out_mat
    return lr_ocr, sr_ocr