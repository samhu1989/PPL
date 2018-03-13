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
                #0,1,2,3,4,5,6,7,8,9,10,11
                [0,1,2,3,4,5,6,7,0,1, 2, 3],
                [1,2,3,0,5,6,7,4,4,5, 6, 7]
                ],dtype=np.int32);
        self.fedge_idx_v = np.array(
                [
                [ 4,5, 6,7],
                [ 8,7,11,3],
                [ 9,4, 8,0],
                [10,5, 9,1],
                [11,6,10,2]
                ]
                ,dtype=np.int32);
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
    
    def get_xy_grid(self):
        xy_grid = (np.transpose(np.mgrid[0:256,0:256],[0,2,1])).astype(np.float32);
        return self.ViewCoordToNormCoord(xy_grid);
    
    def get_inside(self,xy):
        inside = np.zeros([256,256,5],dtype=np.float32);
        x = self.xy_grid_v[0,...];
        y = self.xy_grid_v[1,...];
        for i in range(0,5):
            ABCD = xy[:,self.fidx[i]];
            #print(ABCD)
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
            mask = fequal(a1,a2,1.0/256.0);
            #print(np.sum(mask));
            inside[mask,i] = 1.0;
        return inside;
    
    def get_nn_idx(self,pts):
        nnidx = np.zeros([256,256,5],dtype=np.int32);
        x = self.xy_grid_v.reshape((2,-1));
        x = np.transpose(x,[1,0]);
        for i in range(5):
            fedge_idx = self.fedge_idx_v[i,:];
            fedge_pts = np.transpose(pts[:,:,fedge_idx].reshape(2,-1),[1,0]);
            tree = cKDTree(fedge_pts);
            dist,index = tree.query(x); 
            for j in range(fedge_idx.size):
                start = j*512;
                end = (j+1)*512;
                mask = (index >= start) & ( index < end );
                index[mask] -= start;
                index[mask] += fedge_idx[j]*512;
            nnidx[:,:,0] = index.reshape((256,256));
        return nnidx;
        
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
        self.debug_prob = tf.reduce_sum(self.gt_lbl*(self.max_depth*self.inside + (1.0 - self.inside)*self.dist),axis=2);
        self.prob = tf.nn.softmax(self.max_depth*self.inside + (1.0 - self.inside)*self.dist,dim=-1);
        self.cross_entropy = tf.reduce_sum(tf.negative(self.gt_lbl*tf.log(self.prob)),axis=2,name='cross_entropy');
        self.gt_loss = tf.reduce_mean(self.cross_entropy,name='gt_loss');
        self.out_norm_idx = tf.constant([3],dtype=tf.int32,shape=[1],name="out_norm_idx");
        self.out_norm = tf.gather(self.out,self.out_norm_idx,name="out_norm");
        self.norm_loss = tf.reduce_mean(tf.square(self.out_norm - 1.0),name="norm_loss");
        self.loss = self.gt_loss + 100.0*self.norm_loss;
        
class PPPixWork(QtCore.QObject):
    def __init__(self,parent=None):
        super(PPPixWork,self).__init__(parent);
        self.ppl = PPPix("/cpu:0");
        config = tf.ConfigProto();
        config.allow_soft_placement = True;
        self.sess = tf.Session(config=config);
        self.sess.run(tf.global_variables_initializer());
        self.init_affine = np.array(
                [[ 7.68641138,0.0,0.0,0.0],
                 [ 0.0,-7.68641138,0.0,0.0],
                 [ 0.0,0.0,4.77076721,-0.99086928]]
                );
        self.sess.run(self.ppl.set_affine,feed_dict={self.ppl.extern_affine:self.init_affine});
        self.lrate = 0.001
        self.iter = 0;
        
    def setGT(self,lmat):
        self.gt,self.gt_lbl,_ = self.ppl.get_gt_lbl(lmat);
        
    def getLabel(self):
        out_xy = self.sess.run(self.ppl.out_xy);
        inside = self.ppl.get_inside(out_xy);
        edge_pts = self.sess.run(self.ppl.edge_pts);
        nnidx = self.ppl.get_nn_idx(edge_pts);
        feed = {
                self.ppl.inside:inside,
                self.ppl.nnidx:nnidx
                };
        prob = self.sess.run(self.ppl.prob,feed_dict=feed);
        lbl = np.argmax(prob,axis=-1);
        lbl += 1;
        return lbl.astype(np.uint8);
    
    def getDebug(self):
        out_xy = self.sess.run(self.ppl.out_xy);
        inside = self.ppl.get_inside(out_xy);
        edge_pts = self.sess.run(self.ppl.edge_pts);
        nnidx = self.ppl.get_nn_idx(edge_pts);
        feed = {
                self.ppl.gt_lbl:self.gt,
                self.ppl.inside:inside,
                self.ppl.nnidx:nnidx
                };
        debug = self.sess.run(self.ppl.debug_prob,feed_dict=feed);
        debug -= debug.min();
        debug /= debug.max();
        debug *= 255.0;
        array = debug.astype(np.uint8);
        imgs = [];
        imgs.append(QtGui.QImage(array,array.shape[1],array.shape[0],array.shape[1],QtGui.QImage.Format_Grayscale8));
        dist = self.sess.run(self.ppl.dist,feed_dict=feed);
        for i in range(5):
            debug = dist[:,:,i];
            debug -= debug.min();
            debug /= debug.max();
            debug *= 255.0;
            array = debug.astype(np.uint8);
            imgs.append(QtGui.QImage(array,array.shape[1],array.shape[0],array.shape[1],QtGui.QImage.Format_Grayscale8));
        return imgs;
    
    def optimize(self):
        for i in range(3):
            out_xy = self.sess.run(self.ppl.out_xy);
            inside = self.ppl.get_inside(out_xy);
            edge_pts = self.sess.run(self.ppl.edge_pts);
            nnidx = self.ppl.get_nn_idx(edge_pts);
            feed = {
                self.ppl.lr:self.lrate,
                self.ppl.gt_lbl:self.gt,
                self.ppl.inside:inside,
                self.ppl.nnidx:nnidx
                };
            _,self.loss_value = self.sess.run([self.ppl.opt,self.ppl.loss],feed_dict=feed);
        print(self.loss_value);
        self.iter += 1;
        
class PPPixLabel(QLabel):
    def __init__(self,parent=None):
        super(PPPixLabel,self).__init__(parent);
        self.thread = PPThread(self);
        self.work = PPPixWork();
        self.work.moveToThread(self.thread);
        self.thread.timer.setInterval(1000);
        self.thread.timer.timeout.connect(self.updateimg);
        self.thread.timer.timeout.connect(self.work.optimize);
        self.imgpad = QtGui.QImage(256*4,256*2,QtGui.QImage.Format_RGB888);
        self.imgpad.fill(Qt.black);
        
    def setGT(self,lmat):
        self.work.setGT(lmat);
        lblimg = (convertLabelToQImage(self.work.gt_lbl)).convertToFormat(QtGui.QImage.Format_RGB888);
        painter = QtGui.QPainter();
        painter.begin(self.imgpad);
        painter.drawImage(0,0,lblimg);
        painter.end();
    
    @pyqtSlot()
    def updateimg(self):
        print('updating');
        lbl = self.work.getLabel();
        lblimg = (convertLabelToQImage(lbl)).convertToFormat(QtGui.QImage.Format_RGB888);
        painter = QtGui.QPainter();
        painter.begin(self.imgpad);
        painter.drawImage(256,0,lblimg);
        painter.end();
        debugimg = self.work.getDebug();
        for idx,img in enumerate(debugimg):
            painter.begin(self.imgpad);
            i = idx+2;
            ix = i%4;
            iy = i//4;
            painter.drawImage(256*ix,256*iy,img);
            painter.end();
        self.setPixmap(QtGui.QPixmap.fromImage(self.imgpad));
        self.update();
        
    def start(self):
        self.thread.start();
        
    def closeEvent(self,e):
        self.thread.quit();
        self.thread.wait();
        super(PPPixLabel,self).closeEvent(e);
        
if __name__ == "__main__":
    qapp = QApplication(sys.argv);
    layout = listdir("E:\\WorkSpace\\LSUN\\layout\\layout_seg",".mat");
    lmat = loadmat(layout[9])['layout'].copy();
    lmat[lmat==2]=4;
        #print(xy);
    #img = convertLabelToQImage(inside_lbl.astype(np.uint8));
    qlbl = PPPixLabel();
    qlbl.setGT(lmat);
    qlbl.updateimg();
    #qlbl.setPixmap(QtGui.QPixmap.fromImage(img));
    qlbl.show();
    qlbl.start();
    qapp.exec();
    