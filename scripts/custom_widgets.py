'''
Python script to create custom Widgets:
    1. QGraphicsView that can accept input by drag n drop
    2. SVG button to create the zoom-in zoom-out buttons using the SVGs
'''
from scripts.utils import resource_path
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QPushButton
from PySide6.QtGui import QPixmap, QPainter,  QIcon
from PySide6.QtCore import Qt, QSize, Signal


class ControlView(QGraphicsView):
    drop_signal = Signal(str)
    
    def __init__(self, scene, parent):
        super(ControlView, self).__init__(parent)
        self.setObjectName('ControlView')
        self.setScene(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.input_pixmap = None
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.file_path = None
        self.dragOver = False
    
    def get_file_path(self):
        return self.file_path
    
    def update_file_path(self, path):
        self.file_path = resource_path(path)
        self.input_pixmap = QPixmap(self.file_path)

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()
       
    def dragEnterEvent(self, event):  
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()
           
    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.drop_signal.emit(file_path)
            self.set_scene(file_path)
            self.update_file_path(file_path)
            event.accept()
        else:
            event.ignore()
            
    def get_pixmap(self):
        return self.input_pixmap

    def update_pixmap(self, pixmap):
        self.input_pixmap = pixmap

    def set_scene(self, file_path):
        new_scene = QGraphicsScene()
        self.input_pixmap = QPixmap(file_path)
        new_scene.addPixmap(QPixmap(file_path))
        self.setScene(new_scene)

class SVGButton(QPushButton):
    def __init__(self, svg_path, parent=None):
        super().__init__(parent)
        self.setIcon(QIcon(svg_path))
        self.setIconSize(QSize(32, 32))  # Set the icon size as needed
        self.setFlat(True)  # Make the button flat (no borders)
        self.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.2);  /* Semi-transparent black on hover */
            }
            QPushButton:pressed {
                background-color: rgba(0, 0, 0, 0.4);  /* Even darker color when pressed */
            }
            """
        )