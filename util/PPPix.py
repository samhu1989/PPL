# -*- coding: utf-8 -*-
import sys;
import os;
from PPL import PPAffine;
import tensorflow as tf;
import numpy as np;
from scipy.io import loadmat;
from PyQt5 import QtGui,QtCore,QtWidgets;
from PyQt5.QtCore import Qt;
from PyQt5.QtWidgets import QLabel,QApplication;
from QImage2Array import convertQImageToLabel2;
from QImage2Array import convertLabelToQImage;
from InterpZ import *;
util_path = os.path.dirname(__file__)+os.sep+".."+os.sep+"net";
sys.path.append(util_path);
from data import listdir;

class PPPix(PPAffine):
    def __init__(self,dev):
        super(PPPix,self).__init__(dev);
        
    def __get_depth__(self):
        self.out_z_idx = tf.constant([2],dtype=tf.int32,shape=[1],name="out_z_idx");
        self.out_z = tf.reshape(tf.gather(self.out,self.out_z_idx),[8],name="out_z");
        self.depth_idx_v = np.zeros([256,256,5,4],dtype=np.int32);
        self.depth_idx_v[:,:,0,:] = self.fidx[0,:];
        self.depth_idx_v[:,:,1,:] = self.fidx[1,:];
        self.depth_idx_v[:,:,2,:] = self.fidx[2,:];
        self.depth_idx_v[:,:,3,:] = self.fidx[3,:];
        self.depth_idx_v[:,:,4,:] = self.fidx[4,:];
        self.depth_idx = tf.constant(self.depth_idx_v,shape=[256,256,5,4],dtype=tf.int32,name="depth_idx");
        self.depth = tf.gather(self.out_z,self.depth_idx,name='depth');
        self.max_depth = tf.reduce_max(tf.square(self.depth),axis=3,name='max_depth');
        
    def __get_dist__(self):
        self.edge_idx_v = np.array(
                [
                [0,1,2,3,4,5,6,7,0,1,2,3],
                [1,2,3,0,5,6,7,4,4,5,6,7]
                ],dtype=np.int32);
        self.edge_idx = tf.constant(self.edge_idx_v,dtype=tf.int32,shape=[2,1,12],name="edge_idx");
        self.edge_end = tf.gather(self.out_xy,self.edge_idx,axis=1,name='edge_end');#[2,2,1,12]
        self.interp_w_v = np.zeros([2,512],dtype=np.float32);
        self.interp_w_v[0,:] = np.linspace(0.0,1.0,512,dtype=np.float32);
        self.interp_w_v[1,:] = 1.0 - self.interp_w_v[1,:];
        self.interp_w = tf.constant(self.interp_w_v,dtype=tf.float32,shape=[1,2,512,1]);
        self.edge_pts = tf.reduce_sum(self.interp_w*self.edge_end,axis=1,name='edge_pts');
        self.edge_pts_flat = tf.reshape(self.edge_pts,[2,-1],name='edge_pts_flat');
        self.nnidx = tf.placeholder(tf.int32,shape=[256,256,5],name='nnidx');
        self.nnp = tf.gather(self.edge_pts_flat,self.nnidx,axis=1,name='nnp');
        self.dist = -tf.reduce_sum( tf.square( self.xy_grid - self.nnp ),axis=0,name='dist');
        
    def get_gt_lbl(self,msk):
        mskimg = QtGui.QImage(msk,msk.shape[1],msk.shape[0],msk.shape[1],QtGui.QImage.Format_Indexed8);
        mskimg = mskimg.scaled(256,256,Qt.KeepAspectRatio);
        lbl = convertQImageToLabel2(mskimg);
        gt_lbl = np.zeros([256,256,5],dtype=np.float32);
        for l in range(1,lbl.max()+1):
            gt_lbl[lbl==l,l-1] = 1.0;
        arglbl = np.argmax(gt_lbl, axis=-1);
        return gt_lbl,lbl,arglbl.astype(np.uint8);
    
    def get_nnidx(self,edge_pts,xy_grid):
        return;
    
    def get_xy_grid(self):
        xy_grid = (np.transpose(np.mgrid[0:256,0:256],[0,2,1])).astype(np.float32);
        return self.ViewCoordToNormCoord(xy_grid);
    
    def get_inside(self,xy):
        inside = np.zeros([256,256,5]);
        x = self.xy_grid_v[0,...];
        y = self.xy_grid_v[1,...];
        for i in range(5):
            ABCD = xy[:,self.fidx[i]];
            A = ABCD[:,0];
            B = ABCD[:,1];
            C = ABCD[:,2];
            D = ABCD[:,3];
            aAMB = areaFromV(A,B,x,y);
            aBMC = areaFromV(B,C,x,y);
            aCMD = areaFromV(C,D,x,y);
            aDMA = areaFromV(D,A,x,y);
            aABC = areaFromV(A,B,C[0],C[1]);
            aCDA = areaFromV(A,D,C[0],C[1]);
            a1 = (aAMB+aBMC+aCMD+aDMA) ;
            a2 = (aABC+aCDA);
            mask = fequal(a1,a2,0.99);
            inside[mask,i] = 1.0;
            
        
    def ViewCoordToNormCoord(self,coord):
        newcoord = coord.copy();
        newcoord -= 128.0;
        newcoord[1,:] *= -1.0;
        newcoord /= (256.0/2.0);
        return newcoord;
        
    def __get_loss__(self):
        self.fidx = np.array(
            [[4,5,6,7],
             [0,4,7,3],
             [1,5,4,0],
             [2,6,5,1],
             [3,7,6,2]],dtype=np.int32);
        self.gt_lbl = tf.placeholder(tf.float32,shape=[256,256,5],name='gt_lbl');
        self.xy_grid_v = self.get_xy_grid();#2,256,256]
        self.xy_grid = tf.constant(self.xy_grid_v,dtype=tf.float32,shape=[2,256,256,1],name='xy_grid');
        self.inside = tf.placeholder(tf.float32,shape=[256,256,5],name='inside');
        #probability depend on neareast distance if outside the face
        self.__get_dist__();
        #probability depend max depth if inside the face
        self.__get_depth__();
        #prob
        self.prob =  tf.nn.softmax(self.max_depth*self.inside + (1.0 - self.inside)*self.dist,dim=-1);
        print(self.prob.shape)
        self.cross_entropy = tf.reduce_sum(tf.negative(self.gt_lbl*tf.log(self.prob)),axis=2,name='cross_entropy');
        self.gt_loss = tf.reduce_mean(self.cross_entropy,name='gt_loss');
        self.out_norm_idx = tf.constant([3],dtype=tf.int32,shape=[1],name="out_norm_idx");
        self.out_norm = tf.gather(self.out,self.out_norm_idx,name="out_norm");
        self.norm_loss = tf.reduce_mean(tf.square(self.out_norm - 1.0),name="norm_loss");
        self.loss = self.gt_loss + 100.0*self.norm_loss;
        
if __name__ == "__main__":
    qapp = QApplication(sys.argv);
    ppl = PPPix("/cpu:0");
    layout = listdir("E:\\WorkSpace\\LSUN\\layout\\layout_seg",".mat");
    lmat = loadmat(layout[9])
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    msk = lmat['layout'].copy();
    gt_lbl,_,arglbl = ppl.get_gt_lbl(msk);
    img = convertLabelToQImage(arglbl);
    qlbl = QLabel();
    qlbl.setPixmap(QtGui.QPixmap.fromImage(img));
    qlbl.show();
    qapp.exec();
    