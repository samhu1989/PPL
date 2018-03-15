# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:43:18 2018

@author: SamHu
"""
from PyQt5 import QtWidgets;
from PyQt5.QtWidgets import QDialog;
from PyQt5.uic import loadUi;
class SpinDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()     
        loadUi('./spindialog.ui',self);