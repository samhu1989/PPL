# -*- coding: utf-8 -*-
import sys;
import os;
from PPL import PPAffine;
import tensorflow as tf;
import numpy as np;
from scipy.io import loadmat;
from PyQt5 import QtGui,QtCore,QtWidgets;
from PyQt5.QtCore import Qt,pyqtSignal,pyqtSlot;
from PyQt5.QtWidgets import QLabel,QApplication;
from QImage2Array import convertQImageToLabel2;
from QImage2Array import convertLabelToQImage;
from InterpZ import areaFromV;
from InterpZ import fequal;
from scipy.spatial import cKDTree;
from PPLUi import PPThread;
path = os.path.dirname(__file__);
if path:
    util_path = path+os.sep+".."+os.sep+"net";
else:
    util_path = ".."+os.sep+"net";
sys.path.append(util_path);
from data import listdir;
from PPNet import PPNet
class PPCorner(PPNet):
    def __init__(self,dev,param):
        super(PPCorner,self).__init__(dev,param);
        
        
    def __get_loss__(self):