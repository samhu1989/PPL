# -*- coding: utf-8 -*-
from PPL import PPAffine;
import tensorflow as tf;
import numpy as np;

class PPPix(PPAffine):
    def __init__(self,dev):
        super(PPPix,self).__init__(dev);
        
    def __get_depth_idx__(self):
        self.depth_idx_v = np.zeros([256,256,5,4],dtype=np.int32);
        self.depth_idx_v[:,:,0,:] = self.fidx[0,:];
        self.depth_idx_v[:,:,1,:] = self.fidx[1,:];
        self.depth_idx_v[:,:,2,:] = self.fidx[2,:];
        self.depth_idx_v[:,:,3,:] = self.fidx[3,:];
        self.depth_idx_v[:,:,4,:] = self.fidx[4,:];
        self.depth_idx = tf.constant(self.depth_idx_v,shape=[256,256,5,4],dtype=tf.int32,name="depth_idx");
        
    def __get_loss__(self):
        self.fidx = np.array(
            [[4,5,6,7],
             [0,4,7,3],
             [1,5,4,0],
             [2,6,5,1],
             [3,7,6,2]],dtype=np.int32);
        self.gt_lbl = tf.placeholder(tf.float32,shape=[256,256,5],name='gt_lbl');
        self.xy_grid_v = np.transpose(np.mgrid[0:256,0:256],[0,2,1]);#[2,256,256]
        self.xy_grid = tf.constant(self.xy_grid_v,dtype=tf.float32,shape=[2,256,256,1],name='xy_grid');
        self.inside = tf.placeholder(tf.float32,shape=[256,256,5],name='inside');
        #probability depend on neareast distance if outside the face
        self.edge_end = #[2,2,1,12]
        
        self.interp_w_v = np.zeros([2,512],dtype=np.float32);
        self.interp_w_v[0,:] = np.linspace(0.0,1.0,512,dtype=np.float32);
        self.interp_w_v[1,:] = 1.0 - self.interp_w_v[1,:];
        self.interp_w = tf.constant(self.interp_w_v,dtype=tf.float32,shape=[1,2,512,1]);
        self.edge_pts = tf.gather;
        self.edge_pts_flat = tf.reshape(self.edge_pts,[2,-1],name='edge_pts_flat');
        self.nnidx = tf.placeholder(tf.int32,shape=[256,256,5],name='nnidx');
        self.nnp = tf.gather(self.edge_pts,self.nnidx,,axis=1,name='nnp');
        self.dist = - tf.reduce_sum( tf.square( self.xy_grid - self.nnp ),axis=0,name='dist');
        #probability depend max depth if inside the face
        self.out_z_idx = tf.constant([2],dtype=tf.int32,shape=[1],name="out_z_idx");
        self.out_z = tf.gather(self.out,self.out_z_idx,name="out_z");
        self.__get_depth_idx__();
        self.depth = tf.gather(self.out_z,self.depth_idx,name='depth');
        self.max_detph = tf.reduce_max(tf.square(self.depth),axis=3,name='max_depth');
        #prob
        self.prob =  tf.nn.softmax(self.max_depth*self.inside + (1.0 - self.inside)*self.dist,axis=2);
        self.cross_entropy = tf.reduce_sum(tf.negative(self.gt_lbl*tf.log(self.prob,axis=2)),axis=2,name='cross_entropy');
        self.gt_loss = tf.reduce_mean(self.cross_entropy,name='gt_loss');
        self.out_norm_idx = tf.constant([3],dtype=tf.int32,shape=[1],name="out_norm_idx");
        self.out_norm = tf.gather(self.out,self.out_norm_idx,name="out_norm");
        self.norm_loss = tf.reduce_mean(tf.square(self.out_norm - 1.0),name="norm_loss");
        self.loss = self.gt_loss + 100.0*self.norm_loss;