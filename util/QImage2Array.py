# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 10:37:48 2018

@author: SamHu
"""
import numpy as np;
from PyQt5 import QtGui;

def convertQImageToArray(incomingImage):
    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format_RGB32);
    width = incomingImage.width();
    height = incomingImage.height();
    ptr = incomingImage.bits();
    ptr.setsize(incomingImage.byteCount());
    arr = np.array(ptr).reshape(height, width, 4);
    arr = arr[:,:,0:3];
    arr = arr.astype(np.float32);
    arr /= 255.0;
    return arr;

def convertArrayToQImage(array):
    array *= 255.0;
    array = array.astype(np.uint8);
    img = QtGui.QImage(array,array.shape[1],array.shape[0],array.shape[1]*array.shape[2],QtGui.QImage.Format_RGB888);
    return img;