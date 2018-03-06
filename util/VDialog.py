from PyQt5 import QtWidgets;
from PyQt5.QtWidgets import QDialog;
from PyQt5.uic import loadUi;

class VDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()     
        loadUi('./vdialog.ui',self);
        
    def setPixmap(self,pixmap):
        self.label.setPixmap(pixmap);
        