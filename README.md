# Super-Resolution and OCR GUI

## Overview

This project provides a PySide6-based GUI that enables users to study the effects of Super-Resolution on Optical Character Recognition (OCR) results. Users can select an image, choose a Super-Resolution model, select an OCR model, and run the process. The application first applies the Super-Resolution model to the input image and then performs OCR on both the original and super-resolved images. The results are displayed side-by-side for easy comparison.

## Features

- **Super-Resolution Models**: Choose from a variety of non-diffusion and diffusion-based Super-Resolution models.
- **OCR Models**: Select from multiple OCR models.
- **Result Display**: View and compare input images, super-resolved images, and their respective OCR outputs.
- **QGraphicsView for Display**: Use QGraphicsView to avoid resizing images, allowing pixel-to-pixel comparison.
- **Image Switching**: Toggle between the input image and its OCR output, as well as the super-resolved image and its OCR output.
- **Status Bar**: Monitor the time taken by the Super-Resolution and OCR models.
- **Save Options**: Save images in various formats:
  1. Super-resolved Output
  2. Super-resolved OCR Output
  3. OCR Input
  4. 1x4 Input Output (input and super-resolved images in their original pixel format)
  5. 4x4 Input Output (input image duplicated for comparison)
- **Drag and Drop**: Easily upload images by dragging and dropping them into the application.
- **Zoom Controls**: Zoom in and out on images for detailed inspection.

## Future Work
- [ ] Include more diffusion-based Super-Resolution models.
- [ ] Add new scene-based OCR models to enhance functionality.

## Installation

To install and run the project, follow these steps:

1. **Clone the repository**:
   ```
   git clone https://github.com/VinatG/SuperResolution-OCR-GU
   cd SuperResolution-OCR-GUI
   ```
   
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Download the `ONNX_models` and `Tesseract_OCR` folders from the provided Google Drive links:
    - [Download ONNX_models](https://drive.google.com/drive/folders/1ldVCpX-sMke2SJi5yXpVtbXfa5s13h0Y?usp=sharing)
    - [Download Tesseract_OCR](https://drive.google.com/drive/folders/1QimCAPxLnpvBfEwo-ZqUE-_XmMZiTfdD?usp=sharing)

4. Extract and place the `ONNX_models` and `Tesseract_OCR` folders into the main project directory:
    ```bash
    Super-Resolution-OCR-GUI/
    ├── ONNX_models/
    ├── Tesseract_OCR/
    ├── main.py
    └── ...
    ```
5. Run the application:
```
python app.py
```

## Usage
- Launch the application.
- Select an image to process.
- Choose a Super-Resolution model' Use the left dropdown menu for non-diffusion-based SR models and the right dropdown for diffusion-based models.
- Choose an OCR model from the dropdown menu.
- Click the "Run" button to start the process.
- View and compare the results using the provided options.
- Save the images in the desired format using the save options.

## Supported Models
### Non-Diffusion Based Super-Resolution Models
- **[A2N](https://github.com/haoyuc/A2N)**
- **[DAN](https://github.com/greatlog/RealDAN/tree/main)**
- **[ESRGAN](https://github.com/xinntao/Real-ESRGAN)**
- **[SRNO](https://github.com/2y7c3/Super-Resolution-Neural-Operator)**
- **[LIIF](https://github.com/yinboc/liif)**
- **[Swin2SR](https://huggingface.co/docs/transformers/model_doc/swin2sr)**
- **[PAN](https://github.com/zhaohengyuan1/PAN)**
- **[HAN](https://github.com/wwlCape/HAN)**
- **[DRLN](https://github.com/saeed-anwar/DRLN)**
- **[MDBN](https://github.com/thy960112/MDBN/tree/main)**
- **[FEMASR](https://github.com/chaofengc/FeMaSR/tree/main)**
- **[HAT](https://github.com/XPixelGroup/HAT/tree/main)**

### Diffusion-Based Super-Resolution Models
- **[DiffIR](https://github.com/Zj-BinXia/DiffIR)**

### OCR Models
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**
- **PyTesseract**

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your improvements.


