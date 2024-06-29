from PySide6.QtWidgets import QCheckBox, QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QGraphicsScene, QGraphicsView, QVBoxLayout, QHBoxLayout, QWidget,QComboBox
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtCore import Qt
from time import time
import qimage2ndarray
from pathlib import Path
import numpy as np
import cv2
from scripts import execute_sr, execute_ocr,  utils, custom_widgets
import onnxruntime as rt
import sys

#Loading the json file containing the mapping 'model_name':(github,paper)
mapping_dictionary = utils.load_json_file('files\\model_to_links_mapping.json') #Loading the json file
SISR_mapping = mapping_dictionary['non_diffusion_image_super_resolution'] #Extracting the dictionary containing the mapping for non diffusion based ISR models
DSISR_mapping= mapping_dictionary['diffusion_image_super_resolution'] #Extracting the dictionary containing the mapping for diffusion based ISR models
OCR_mapping= mapping_dictionary['OCR'] 
ISR_links_mapping={} #Combining the Super-Resolution mappings into one
ISR_links_mapping.update(SISR_mapping)
ISR_links_mapping.update(DSISR_mapping)


class SuperResolutionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.lr_ocr = None #Output of OCR performed on the Input image
        self.sr_ocr = None #Output of OCR performed on the output of the SR model
        self.sess = None #Loading of the onnx session
        self.checkbox_state = False 
        self.output_image = None #To store the output of the SR model

        self.current_sr_model = 'a2n' #Default SR model
        self.current_ocr_model = 'paddleocr' #Default OCR model

        self.time_taken_SR = -1
        self.time_taken_OCR = -1
        self.input_zoom_factor = 1.0
        self.output_zoom_factor = 1.0
        self.provider = ''

        self.output_pixmap = None
        self.input_non_ocr_pixmap = None
        self.input_ocr_pixmap = None
        self.output_non_ocr_pixmap = None
        self.output_ocr_pixmap = None

        self.input_output_1x4_pixmap = None
        self.input_output_4x4_pixmap = None
        self.github_link_text='<a href="https://github.com/xinntao/Real-ESRGAN">GitHub Repository</a>'
        self.paper_link_text='<a href="https://arxiv.org/pdf/1809.00219.pdf">Research Paper</a>'        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Super Resolution GUI")
        self.setGeometry(100, 100, 800, 600)

        #Viewer->input
        self.input_scene = QGraphicsScene()
        self.input_view = custom_widgets.ControlView(self.input_scene, self) # QGraphic Scene improvised to accept drag & drop inputs
        self.input_view.drop_signal.connect(self.set_input_view)
        self.output_scene = QGraphicsScene()
        self.output_view = QGraphicsView(self.output_scene, self)

        # Create a QPushButton for saving the image
        self.save_image_button = QPushButton('Save Image', self)
        self.save_image_button.clicked.connect(self.save_image)

        # QPushButton to select the image
        self.select_image_button = QPushButton("Select Image", self)
        self.select_image_button.clicked.connect(self.select_image)

        # Drop down to provide the various saving options to the user
        self.save_drop_down = QComboBox(self)
        self.save_drop_down.addItem('SupeResolved Output')
        self.save_drop_down.addItem('SuperResolved OCR Output')
        self.save_drop_down.addItem('OCR Input')
        self.save_drop_down.addItem('1x4 input output')
        self.save_drop_down.addItem('4x4 input output')
        
        # Putting the select image button and the save image button in one row
        self.select_save_buttons = QHBoxLayout()
        self.select_save_buttons.addWidget(self.select_image_button)
        self.select_save_buttons.addWidget(self.save_drop_down)
        self.select_save_buttons.addWidget(self.save_image_button)

        # Labels to the GitHub and the paper link
        self.github_link = QLabel(self.github_link_text)
        self.github_link.setOpenExternalLinks(True)
        self.paper_link = QLabel(self.paper_link_text)
        self.paper_link.setOpenExternalLinks(True)

        # Status Bar where we display the time taken to do the SR and the OCR processing
        self.statusBar().showMessage("Select an image to process.")
        
    #DROP DOWNS

        # Drop-down to select a non-diffusion based SR model
        self.SISR_models_drop_down = QComboBox(self)
        self.DSISR_models_drop_down = QComboBox(self)
        for i in SISR_mapping.keys():
            self.SISR_models_drop_down.addItem(i)
            
        # Drop-down to select a diffusion based SR model
        for i in DSISR_mapping.keys():
            self.DSISR_models_drop_down.addItem(i)
        self.SISR_models_drop_down.currentTextChanged.connect(self.change_links)
        self.SISR_models_drop_down.currentTextChanged.connect(self.change_sr_model)
        self.DSISR_models_drop_down.currentTextChanged.connect(self.change_links)
        self.DSISR_models_drop_down.currentTextChanged.connect(self.change_sr_model)

        # Drop-down to select the OCR model
        self.ocr_drop_down = QComboBox(self)
        for i in OCR_mapping.keys():
            self.ocr_drop_down.addItem(i)
        self.ocr_drop_down.currentTextChanged.connect(self.change_ocr_model)

        # Drop Down to select the provider on which the model will be run
        self.providers_drop_down = QComboBox(self)
        providers_list=rt.get_available_providers()
        self.provider=providers_list[-1]
        for i in reversed(providers_list):
            self.providers_drop_down.addItem(i)
        self.providers_drop_down.currentTextChanged.connect(self.change_providers)

        # Execute button to run the SR + OCR model
        self.run = QPushButton("Run")
        self.run.clicked.connect(self.run_execution)

        # Horizontal Box layout to combine the links together
        self.hbox_links=QHBoxLayout()
        self.hbox_links.addWidget(self.github_link)
        self.hbox_links.addWidget(self.paper_link)

        # Buttons to perform zoom-in and zoom-out
        self.zoom_in_action = QAction('Zoom In', self)
        self.zoom_in_action.triggered.connect(self.input_zoomIn)
        self.zoom_in_action.triggered.connect(self.output_zoomIn)
        self.zoom_out_action = QAction('Zoom Out', self)
        self.zoom_out_action.triggered.connect(self.input_zoomOut)
        self.zoom_out_action.triggered.connect(self.output_zoomOut)

        # Checkbox to apply the zoom on both the images together
        self.checkbox = QCheckBox('Apply Zoom together')
        self.checkbox.stateChanged.connect(self.checkbox_state_changed)

        # Zoom buttons for the Input side
        self.input_zoomButtons = QHBoxLayout()
        self.input_zoom_in = custom_widgets.SVGButton(utils.resource_path('files\\zoom_in.svg'))
        self.input_zoom_in.clicked.connect(self.input_zoomIn)
        self.input_zoom_out = custom_widgets.SVGButton(utils.resource_path('files\\zoom_out.svg'))
        self.input_zoom_out.clicked.connect(self.input_zoomOut)
        self.input_zoomButtons.addStretch(1)
        self.input_zoomButtons.addWidget(self.input_zoom_in)
        self.input_zoomButtons.addWidget(self.input_zoom_out)
        self.input_zoomButtons.addStretch(1)

        # Zoom buttons for the Output side
        self.output_zoomButtons = QHBoxLayout()
        self.output_zoom_in = custom_widgets.SVGButton(utils.resource_path('files\\zoom_in.svg'))
        self.output_zoom_in.clicked.connect(self.output_zoomIn)
        self.output_zoom_out = custom_widgets.SVGButton(utils.resource_path('files\\zoom_out.svg'))
        self.output_zoom_out.clicked.connect(self.output_zoomOut)
        self.output_zoomButtons.addStretch(1)
        self.output_zoomButtons.addWidget(self.output_zoom_in)
        self.output_zoomButtons.addWidget(self.output_zoom_out)
        self.output_zoomButtons.addStretch(1)
        
        # Combining the Input view and the Output view
        self.images = QHBoxLayout()
        self.images.addWidget(self.input_view)
        self.images.addWidget(self.output_view)
        self.buttons = QHBoxLayout()
        self.buttons.addLayout(self.input_zoomButtons)
        self.buttons.addWidget(self.checkbox)
        self.buttons.addLayout(self.output_zoomButtons)

        # Combining the SR models drop-downs
        self.sr_model_drop_downs = QHBoxLayout()
        self.sr_model_drop_downs.addWidget(self.SISR_models_drop_down)
        self.sr_model_drop_downs.addWidget(self.DSISR_models_drop_down)

        # Drop dowm to select whether to display the Normal input image or the OCR image
        self.input_image_drop_down = QComboBox(self)
        self.input_image_drop_down.addItem('Normal Image')
        self.input_image_drop_down.addItem('OCR Image')
        self.input_image_drop_down.currentTextChanged.connect(self.change_input_pixmap)

        # Drop dowm to select whether to display the super-resolved image or the super-resolved + OCR image
        self.output_image_drop_down = QComboBox(self)
        self.output_image_drop_down.addItem('Normal Image')
        self.output_image_drop_down.addItem('OCR Image')
        self.output_image_drop_down.currentTextChanged.connect(self.change_output_pixmap)
        
        # Combiing the drop downs 
        self.single_drop_down = QHBoxLayout()
        self.single_drop_down.addWidget(self.input_image_drop_down)
        self.single_drop_down.addWidget(self.output_image_drop_down)
        
        # Combining the OCR drop down and the providers drop down
        self.line2_drop_down = QHBoxLayout()
        self.line2_drop_down.addWidget(self.ocr_drop_down)
        self.line2_drop_down.addWidget(self.providers_drop_down)
        
        # QLabels to display the current magnification levels
        self.input_magnification_label = QLabel(self)
        self.output_magnification_label = QLabel(self)
        self.input_magnification_label.setText(f'Current Input Magnification Level : {round(self.input_zoom_factor, 3)}')
        self.output_magnification_label.setText(f'Current Output Magnification Level : {round(self.output_zoom_factor, 3)}')
        self.magnification_labels = QHBoxLayout()
        self.magnification_labels.addWidget(self.input_magnification_label)
        self.magnification_labels.addWidget(self.output_magnification_label)
        
        # Vertical Box layout to combine all the Horizontal Box Layouts
        vbox_main = QVBoxLayout()
        vbox_main.addLayout(self.sr_model_drop_downs)
        vbox_main.addLayout(self.line2_drop_down)
        vbox_main.addLayout(self.select_save_buttons)
        vbox_main.addWidget(self.run)
        vbox_main.addLayout(self.single_drop_down)
        vbox_main.addLayout(self.images)
        vbox_main.addLayout(self.buttons)
        vbox_main.addLayout(self.hbox_links)
        vbox_main.addLayout(self.magnification_labels)
        central_widget = QWidget(self)
        central_widget.setLayout(vbox_main)
        self.setCentralWidget(central_widget)


    def checkbox_state_changed(self, state):
        if state == 2:  # Checked
            self.checkbox_state = True
        else:
            self.checkbox_state = False

    def save_image(self):
        # Save Name format : <original filename>_<SR model used>_<OCR model used>_<type of image being saved>
        save_name = Path(self.input_view.get_file_path()).stem + '_' + self.current_sr_model + '_' + self.current_ocr_model + '_' + self.save_drop_down.currentText()
        image_path, _ = QFileDialog.getSaveFileName(self, "Save Image", save_name, "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)")  
        pixmap_d = {'SupeResolved Output' : self.output_non_ocr_pixmap,
                    'SuperResolved OCR Output' : self.output_ocr_pixmap,
                    'OCR Input' : self.input_ocr_pixmap,
                    '1x4 input output' : self.input_output_1x4_pixmap,
                    '4x4 input output' : self.input_output_4x4_pixmap}
        if image_path:
            pixmap = pixmap_d[self.save_drop_down.currentText()]
            pixmap.save(image_path)

    def input_zoomIn(self):
        if self.input_view.get_pixmap():
            if self.checkbox_state:
                self.output_zoom_factor *= 2
                self.updateOutputZoomedImage()
            self.input_zoom_factor *= 2
            self.updateInputZoomedImage()

    def input_zoomOut(self):
        if self.input_view.get_pixmap():
            if self.checkbox_state:
                self.output_zoom_factor /= 2
                self.updateOutputZoomedImage()
            self.input_zoom_factor /= 2
            self.updateInputZoomedImage()
            
    def output_zoomIn(self):
        if self.output_pixmap:
            if self.checkbox_state:
                self.input_zoom_factor *= 2
                self.updateInputZoomedImage()
            self.output_zoom_factor *= 2
            self.updateOutputZoomedImage()

    def output_zoomOut(self):
        if self.output_pixmap:
            if self.checkbox_state:
                self.input_zoom_factor /= 2
                self.updateInputZoomedImage()
            self.output_zoom_factor /= 2
            self.updateOutputZoomedImage()

    def updateInputZoomedImage(self):
        if self.input_view.get_pixmap():
            if self.input_zoom_factor == 1.0:
                self.input_zoomed_image_pixmap = self.input_view.get_pixmap()
            else :
                array = utils.zoom(utils.pixmap_to_numpy(self.input_view.get_pixmap()), self.input_zoom_factor)
                array = array[:, :, ::-1]
                self.input_zoomed_image_pixmap =QPixmap.fromImage(qimage2ndarray.array2qimage(array))
            self.display_pixmaps()
        
    def updateOutputZoomedImage(self):
        if self.output_pixmap:
            if self.output_zoom_factor == 1.0:
                self.output_zoomed_image_pixmap = self.output_pixmap
            else :
                array = utils.zoom(utils.pixmap_to_numpy(self.output_pixmap), self.output_zoom_factor)
                array = array[:, :, ::-1] #Convert RGB image to BGR
                self.output_zoomed_image_pixmap = QPixmap.fromImage( qimage2ndarray.array2qimage(array))
            self.display_pixmaps()

    def display_pixmaps(self):
        self.input_scene.clear()
        self.output_scene.clear()
        self.input_scene.addPixmap(self.input_zoomed_image_pixmap)
        self.input_view.setScene(self.input_scene)
        self.input_view.setAlignment(Qt.AlignCenter)
        self.input_magnification_label.setText(f'Current Input Magnification Level : {round(self.input_zoom_factor, 3)}')  
        self.output_scene.addPixmap(self.output_zoomed_image_pixmap)
        self.output_view.setScene(self.output_scene)
        self.output_view.setAlignment(Qt.AlignCenter)
        self.output_magnification_label.setText(f'Current Output Magnification Level : {round(self.output_zoom_factor, 3)}')

    # Function to reset all the pixmaps
    def reset_pixmaps(self):
        self.output_pixmap =QPixmap()
        self.input_non_ocr_pixmap = QPixmap()
        self.input_ocr_pixmap = QPixmap()
        self.output_non_ocr_pixmap = QPixmap()
        self.output_ocr_pixmap = QPixmap()
        self.input_output_1x4_pixmap = QPixmap()
        self.input_output_4x4_pixmap = QPixmap()
        self.input_zoomed_image_pixmap = QPixmap()
        self.output_zoomed_image_pixmap = QPixmap()
        self.input_view.update_pixmap(QPixmap())
        self.input_zoom_factor = 1.0
        self.output_zoom_factor = 1.0

    # Function to reset the drop-downs and set the views
    def set_input_view(self, file_name):   
        self.reset_pixmaps()
        self.input_image_drop_down.setCurrentText('Normal Image') 
        self.output_image_drop_down.setCurrentText('Normal Image') 
        self.input_view.update_file_path(file_name)
        self.input_non_ocr_pixmap = self.input_view.get_pixmap().copy()
        self.input_zoomed_image_pixmap = self.input_view.get_pixmap().copy()   
        self.input_scene.addPixmap(self.input_view.get_pixmap()) 
        self.display_pixmaps()

    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options = options)
        if file_name:
            self.set_input_view(file_name)
            
    def change_providers(self, text):
        self.provider = text
        try:
            self.sess = rt.InferenceSession(utils.model_path(self.current_sr_model), providers = [ self.provider])
        except:
            self.sess = rt.InferenceSession(utils.model_path(self.current_sr_model), providers = ['CPUExecutionProvider'])

    
    def change_output_pixmap(self, text):
        if text == 'Normal Image':
            self.output_pixmap = self.output_non_ocr_pixmap
            self.output_zoomed_image_pixmap = self.output_non_ocr_pixmap.copy()
        else:
            self.output_pixmap = self.output_ocr_pixmap
            self.output_zoomed_image_pixmap = self.output_ocr_pixmap.copy()
        self.output_zoom_factor = 1.0
        self.output_magnification_label.setText(f'Current Output Magnification Level : {round(self.output_zoom_factor, 3)}')  
        self.display_pixmaps()
        
    def change_input_pixmap(self, text):
        if text == 'Normal Image':
            self.input_view.update_pixmap(self.input_non_ocr_pixmap)
            self.input_zoomed_image_pixmap = self.input_non_ocr_pixmap.copy()
        else:
            self.input_view.update_pixmap(self.input_ocr_pixmap)
            self.input_zoomed_image_pixmap = self.input_ocr_pixmap.copy()
        self.input_zoom_factor = 1.0
        self.input_magnification_label.setText(f'Current Input Magnification Level : {round(self.input_zoom_factor, 3)}')  
        self.display_pixmaps()
        
    def change_links(self, text):
        self.github_link_text, self.paper_link_text = utils.cvt_to_link(ISR_links_mapping[text][0], 0), utils.cvt_to_link(ISR_links_mapping[text][1], 1)
        self.github_link.setText(self.github_link_text)
        self.paper_link.setText (self.paper_link_text)
        self.hbox_links.update()
        
    def change_sr_model(self, text):
        if text == '<Not Selected>':
            return
        
        self.current_sr_model = text.lower()
        l_SISR=[i.lower() for i in SISR_mapping.keys()]
        l_DSISR=[i.lower() for i in DSISR_mapping.keys()]
        
        if self.current_sr_model in l_SISR:
            self.DSISR_models_drop_down.setCurrentText('<Not Selected>')

        elif self.current_sr_model in l_DSISR:
            self.SISR_models_drop_down.setCurrentText('<Not Selected>')
            
        try:
            self.sess = rt.InferenceSession(utils.model_path(self.current_sr_model), providers = [self.provider])
        except:
            self.sess=rt.InferenceSession(utils.model_path(self.current_sr_model), providers = ['CPUExecutionProvider'])
    
    
    def change_ocr_model(self, text):
        self.current_ocr_model = text.lower()
    
    
    def img2nmp(self):
        image = cv2.imread(self.input_view.get_file_path())
        image_array = np.asarray(image, dtype = np.float32) / 255.0
        image_array = image_array[..., ::-1]
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, 0)
        return image_array
    
    def output_on_pad_image(self, sess, output_name, inputs, window_size):
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = inputs.shape
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        pad_width = ((0, 0), (0, 0), (0, mod_pad_h), (0, mod_pad_w))
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
    
    def repeat_pixels(self, original_image):
        return np.repeat(np.repeat(original_image, 4, axis=0), 4, axis=1)

    def run_execution(self):
        # Load the session in case it is not already loaded
        self.input_zoom_factor = 1.0
        self.output_zoom_factor = 1.0
        if self.sess is None:
            try:
                self.sess = rt.InferenceSession(utils.model_path(self.current_sr_model), providers = [self.provider])
            except:
                self.sess = rt.InferenceSession(utils.model_path(self.current_sr_model), providers = ['CPUExecutionProvider'])
        
        # Computing the Super-Resolved image
        start_time = time()
        sr_out_mat = execute_sr.execute_sr(self.current_sr_model, self.sess, self.input_view.get_file_path())
        inp_img_array = (self.img2nmp() * 255.0).squeeze(axis = 0)
        self.output_non_ocr_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(sr_out_mat))
        end_time = time()
        self.time_taken_SR = end_time - start_time # Time taken by the SR model

        # Performing OCR
        start_time = time()
        h, w, c = sr_out_mat.shape
        white_image_1x4 = np.ones((h, w * 2, c), dtype = sr_out_mat.dtype) * 255.0
        white_image_4x4 = np.ones((h, w * 2, c), dtype = sr_out_mat.dtype) * 255.0
        inp_img_array = inp_img_array.transpose(1, 2, 0)
        white_image_1x4[0 : inp_img_array.shape[0], 0 : inp_img_array.shape[1], :] = inp_img_array
        white_image_1x4[: , w :, :] = sr_out_mat
    
        r_inp_img_aray = self.repeat_pixels(inp_img_array) # 4x Resized input image using duplication for comparison
        white_image_4x4[:, 0 : w, :] = r_inp_img_aray
        white_image_4x4[: , w :, :] = sr_out_mat
        self.input_output_1x4_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(white_image_1x4))
        self.input_output_4x4_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(white_image_4x4))

        inp_img_array = np.ascontiguousarray(inp_img_array, dtype = np.uint8)
        sr_out_mat = np.ascontiguousarray(sr_out_mat, dtype = np.uint8)
        
        #Performing OCR on both the input image and the output SR image
        lr_ocr, sr_ocr = execute_ocr.execute_ocr(self.current_ocr_model, sr_out_mat, inp_img_array)
        end_time = time()
        self.time_taken_OCR = end_time - start_time
        
        # Updating the pixmaps
        sr_ocr_pix = QPixmap.fromImage(qimage2ndarray.array2qimage(sr_ocr))
        lr_ocr_pix = QPixmap.fromImage(qimage2ndarray.array2qimage(lr_ocr))
        self.output_ocr_pixmap = sr_ocr_pix.copy()
        self.input_ocr_pixmap = lr_ocr_pix.copy()
        self.output_pixmap = sr_ocr_pix.copy()
        self.input_view.update_pixmap(lr_ocr_pix.copy())
        self.output_zoomed_image_pixmap = sr_ocr_pix.copy()
        self.input_zoomed_image_pixmap = lr_ocr_pix.copy()

        # Updating the displays
        self.output_image_drop_down.setCurrentText('OCR Image') 
        self.input_image_drop_down.setCurrentText('OCR Image') 
        # Display the pixmaps and show the time taken
        self.display_pixmaps()
        self.show_time_taken()

    def show_time_taken(self):
        self.statusBar().showMessage(f"Time taken by SR: {self.time_taken_SR:.4f} seconds. Time taken by OCR: {self.time_taken_OCR:.4f} seconds")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    stylesheet = """
    QGraphicsView {
        border: 1px solid #CCCCCC; /* Border style for QGraphicsView */
        border-radius: 1px;
    }
    """
    app.setStyleSheet(stylesheet)
    window = SuperResolutionGUI()
    window.show()
    sys.exit(app.exec())