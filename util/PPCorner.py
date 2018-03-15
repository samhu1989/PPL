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
        self.gt_xy = tf.placeholder(tf.float32,shape=[None,2,8],name='gt_xy');
        self.gt_xy_w = tf.placeholder(tf.float32,shape=[None,1,8],name='gt_xy_w');
        self.gt_xy_dist = tf.reduce_sum(tf.square(self.out_xy - self.gt_xy),axis=1);
        self.gt_xy_loss = tf.reduce_mean(self.gt_xy_w*self.gt_dist,name="gt_loss");
        #
        self.out_norm_idx = tf.constant([3],dtype=tf.int32,shape=[1],name="out_norm_idx");
        self.out_norm = tf.gather(self.out,self.out_norm_idx,name="out_norm",axis=1);
        self.norm_loss = tf.reduce_mean(tf.square(self.out_norm - 1.0),name="norm_loss");
        self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1;
        self.loss = self.gt_loss + self.reg_loss + 100.0*self.norm_loss ;
        #
        self.gt_affine = tf.placeholder(tf.float32,shape=[None,3,4],name="gt_affine");
        self.gt_offset = tf.placeholder(tf.float32,shape=[None,2,1],name="gt_offset");
        self.affine_pre = tf.square( self.affine - self.gt_affine );
        self.offset_pre = tf.square( self.offset - self.gt_offset );
        self.pre_loss = tf.reduce_mean(self.affine_pre) + tf.reduce_mean(self.offset_pre) + 100.0*self.norm_loss;