# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 10:37:48 2018

@author: SamHu
"""
import numpy as np;
from PyQt5 import QtGui;
from PyQt5.QtCore import Qt;

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

mskCTable = [   QtGui.qRgba(216,34,13,180),
                QtGui.qRgba(191,221,163,180),
                QtGui.qRgba(255,233,169,180),
                QtGui.qRgba(240,145,146,180),
                QtGui.qRgba(249,209,212,180),
                QtGui.qRgba(223,181,183,180),
                ];

def convertLabelToQImage(lbl):
    msk = QtGui.QImage(lbl,lbl.shape[1],lbl.shape[0],lbl.shape[1],QtGui.QImage.Format_Indexed8);
    msk.setColorTable(mskCTable);
    return msk;

GrayTable = [ QtGui.qRgb(x,x,x) for x in range(256) ];

def convertQImageToLabel(oldimg):
    assert oldimg.format() == QtGui.QImage.Format_Indexed8;
    img = oldimg.copy();
    img.setColorTable(GrayTable);
    img = img.convertToFormat(QtGui.QImage.Format_Grayscale8);
    w = img.width();
    h = img.height();
    s = 2**np.ceil(np.log2(max(w,h)));
    imgpad = QtGui.QImage(s,s,QtGui.QImage.Format_Grayscale8);
    painter = QtGui.QPainter();
    painter.begin(imgpad);
    painter.drawImage(0,0,img);
    painter.end();
    ptr = imgpad.bits();
    assert imgpad.byteCount() == (imgpad.width()*imgpad.height());
    ptr.setsize(imgpad.byteCount());
    lbl = np.array(ptr,dtype=np.uint8).reshape(imgpad.height(),imgpad.width());
    return lbl[0:h,0:w];

def convertQImageToLabel2(oldimg):
    assert oldimg.format() == QtGui.QImage.Format_Indexed8;
    img = oldimg.copy();
    img.setColorTable(GrayTable);
    img = img.convertToFormat(QtGui.QImage.Format_Grayscale8);
    w = img.width();
    h = img.height();
    s = 2**np.ceil(np.log2(max(w,h)));
    imgpad = QtGui.QImage(s,s,QtGui.QImage.Format_Grayscale8);
    imgpad.fill(Qt.black);
    painter = QtGui.QPainter();
    painter.begin(imgpad);
    painter.drawImage((imgpad.width() - img.width())//2,(imgpad.height() - img.height())//2,img);
    painter.end();
    ptr = imgpad.bits();
    assert imgpad.byteCount() == (imgpad.width()*imgpad.height());
    ptr.setsize(imgpad.byteCount());
    lbl = np.array(ptr,dtype=np.uint8).reshape(imgpad.height(),imgpad.width());
    return lbl;