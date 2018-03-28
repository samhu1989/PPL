# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:20:37 2018

@author: SamHu
"""
from PyQt5.QtCore import Qt,pyqtSignal,pyqtSlot;
from PyQt5 import QtCore, QtGui, QtOpenGL,QtWidgets;
from PyQt5.QtWidgets import QApplication,QFileDialog,QMainWindow,QMessageBox,QDialog;
import tensorflow as tf;
import numpy as np;
import sys;
from PPLUi import PPLWidget,PPWork;
from PPCNet import PPCNet;
class PPWorkAuto(PPWork):
    def __init__(self,parent):
        super(PPWorkAuto,self).__init__(parent);
        
    def __init_ppl__(self):
        with tf.device("/cpu:0"):
            param_shape = [1,22];
            param_init  = tf.constant_initializer(
                 [
                  1.0,0.0,0.0,0.0,
                  0.0,1.0,0.0,0.0,
                  0.0,0.0,1.0,0.0,
                  0.0,0.0,0.0,0.0,
                  0.0,0.0,0.0,0.0,
                  0.0,0.0
                 ],dtype=tf.float32);
            param = tf.get_variable(shape=param_shape,initializer=param_init,trainable=True,name='param');
        self.ppl = PPCNet("/cpu:0",param);
        self.loss_value = 1e5;
        
    def __init_const__(self):
        return;

    def setPPLWidget(self,w):
        assert isinstance(w,PPLWidget),'Invalid PPLWidget';
        self.pplwidget = w;
     
    def setLearningRate(self,lr):
        self.lrate = lr;
        
    def setTargetItems(self,target):
        self.gt_items = target;
        
    def init_gt(self):
        sx = -float(self.pplwidget.imgscaled.width())/float(self.pplwidget.viewSize);
        sy = float(self.pplwidget.imgscaled.height())/float(self.pplwidget.viewSize);
        vC = np.array([[sx,sy]],dtype=np.float32);
        mh = int(self.pplwidget.mskmat.shape[0]);
        mw = int(self.pplwidget.mskmat.shape[1]);
        l0 = self.pplwidget.mskmat[0,0];
        l1 = self.pplwidget.mskmat[0,mw-1];
        l2 = self.pplwidget.mskmat[mh-1,mw-1];
        l3 = self.pplwidget.mskmat[mh-1,0];
        vL = np.array([[l0,l1,l2,l3]],dtype=np.int32);
        self.ppl.init_gt(1,[self.pplwidget.current_type],[self.pplwidget.current_corners],vC,vL);
        return;
        
    def getXY(self):
        feed = self.ppl.get_feed(1);
        xy = self.sess.run(self.ppl.out_xy,feed_dict=feed);
        return xy[0,...]; 
    
    def getXYZ(self):
        XYZ1 = self.sess.run(self.ppl.out_hard);
        offset = self.getOffset();
        XYZ1[0:2,:] += offset;
        return XYZ1[0:3,:];
    
    def getAffine(self):
        return self.sess.run(self.ppl.affine);
    
    def getOffset(self):
        return self.sess.run(self.ppl.offset);
    
    @pyqtSlot()
    def prepareOptimize(self):
        self.init_gt();
    
    @pyqtSlot()
    def optimize(self):
        for i in range(100):
            feed = self.ppl.get_feed(1);
            xy = self.sess.run(self.ppl.out_xy,feed_dict=feed);
            self.ppl.update_gt(xy);
            feed = self.ppl.get_feed(1);
            feed[self.ppl.lr] = self.lrate;
            _,self.loss_value= self.sess.run([self.ppl.opt,self.ppl.loss],feed_dict=feed);
        self.iter += 1;
        
def main():
    app = QApplication(sys.argv);
    work = PPWorkAuto(None);
    w = PPLWidget(work=work);
    w.setWindowTitle('PPLUiAuto');    
    w.show();
    sys.exit(app.exec_());

if __name__ == '__main__':
    main();