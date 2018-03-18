# -*- coding: utf-8 -*-
from PPL import PPAffine;
import tensorflow as tf;
import numpy as np;
class PPNet(PPAffine):
    def __init__(self,dev,param):
        with tf.device( dev ):
            self.__prepare_param__(param);
            self.__get_box__();
            self.__get_deformed_box__();
            self.__get_perspect_mat__();
            self.__get_project__();
            self.__get_viewport_map__();
            self.__get_loss__();
            self.__get_opt__();
            self.__get_sum__();
            
    def __prepare_param__(self,param):
        param = tf.reshape(param,[-1,22]);
        self.batch_size = param.shape[0];
        affine_idx = tf.constant([0,1,2,3,4,5,6,7,8,9,10,11],dtype=tf.int32,shape=[3,4]);
        self.affine = tf.gather(param,affine_idx,axis=1,name="affine");
        offset_idx = tf.constant([12,13],dtype=tf.int32,shape=[2,1]);
        self.offset = tf.gather(param,offset_idx,axis=1,name="offset");
        norm_factor_idx = tf.constant([14,15,16,17,18,19,20,21],dtype=tf.int32,shape=[1,8]);
        self.norm_factor = tf.gather(param,norm_factor_idx,axis=1,name="norm_factor")
        self.homo_const = tf.placeholder(tf.float32,shape=[None,1,4],name='homo_const');
        self.affine_mat = tf.concat([self.affine,self.homo_const],axis=1,name="affine_mat");
    
    def get_homo_const(self,batch_size):
        const = np.zeros(shape=[batch_size,1,4],dtype=np.float32);
        const[:,:,3] = 1.0;
        return const;
            
    def __get_box__(self):
        self.homo_box = tf.placeholder(tf.float32,shape=[None,4,8],name='homo_box');
    
    def get_homo_box(self,batch_size):
        one_box = np.array(
                [[[ 0.1, 0.1,-0.1,-0.1, 0.1, 0.1,-0.1,-0.1],
                 [-0.1, 0.1, 0.1,-0.1,-0.1, 0.1, 0.1,-0.1],
                 [ 0.1, 0.1, 0.1, 0.1,-0.1,-0.1,-0.1,-0.1],
                 [   1,   1,   1,   1,   1,   1,   1,   1]
                ]],dtype=np.float32);
        batch_box = np.zeros(shape=[batch_size,4,8],dtype=np.float32) + one_box;
        return batch_box;
    
    def __get_deformed_box__(self):
        self.homo_deformed_box = tf.matmul(self.affine_mat,self.homo_box,name="homo_deformed_box");
        
    def __get_perspect_mat__(self):
        self.fovAngle = np.pi / 2;
        self.fovFar = 1000.0;
        self.fovNear = 0.1;
        self.fovAspect = 1.0;
        self.perspect_mat = tf.placeholder(tf.float32,shape=[None,4,4],name='perspect_mat');
        
    def get_perspect_mat(self,batch_size):
        cot = 1.0 / np.tan(self.fovAngle/2.0);
        f = self.fovFar;
        n = self.fovNear;
        one_perspect_mat = np.array(
            [[[cot/self.fovAspect,0,0,0],
             [0,cot,0,0],
             [0,0,-(f+n)/(f-n),-2*f*n/(f-n)],
             [0,0,-1.0,0]
            ]],dtype=np.float32);
        batch_perspect_mat = np.zeros(shape=[batch_size,4,4],dtype=np.float32) + one_perspect_mat;
        return batch_perspect_mat;
    
    def __get_project__(self):
        self.project_box = tf.matmul(self.perspect_mat,self.homo_deformed_box,name="project_box");
        self.out = tf.multiply(self.norm_factor,self.project_box,name="out");
        #
        w_idx = tf.constant([3],dtype=tf.int32,shape=[1]);
        self.w = tf.gather(self.project_box,w_idx,axis=1);
        self.out_hard = self.project_box / self.w;
        
    def __get_viewport_map__(self):
        self.ext_const = tf.placeholder(tf.float32,shape=[None,2,1],name='ext_const');
        self.offset_ext = tf.concat([self.offset,self.ext_const],axis=1,name="offset_ext");
        self.out_with_offset = tf.add(self.out,self.offset_ext,name="out_with_offset");
        self.out_xy_idx = tf.constant([0,1],dtype=tf.int32,shape=[2],name="out_xy_idx");
        self.out_xy = tf.gather(self.out_with_offset,self.out_xy_idx,axis=1,name="out_xy");
        self.out_hard_with_offset = tf.add(self.out_hard,self.offset_ext,name="out_hard_with_offset");
        self.out_xy_hard = tf.gather(self.out_hard_with_offset,self.out_xy_idx,axis=1,name="out_xy_hard");
        
    def get_ext_const(self,batch_size):
        return np.zeros(shape=[batch_size,2,1],dtype=np.float32);
        
    def __get_loss__(self):
        self.gt_xy = tf.placeholder(tf.float32,shape=[None,2,8],name='gt_xy');
        self.gt_dist = tf.reduce_sum(tf.square(self.out_xy - self.gt_xy),axis=1);
        self.gt_loss = tf.reduce_mean(self.gt_dist,name="gt_loss");
        self.out_norm_idx = tf.constant([3],dtype=tf.int32,shape=[1],name="out_norm_idx");
        self.out_norm = tf.gather(self.out,self.out_norm_idx,name="out_norm",axis=1);
        self.norm_loss = tf.reduce_mean(tf.square(self.out_norm - 1.0),name="norm_loss");
        self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1;
        self.loss = 10.0*self.gt_loss + self.reg_loss + 1000.0*self.norm_loss ;
        #
        self.gt_affine = tf.placeholder(tf.float32,shape=[None,3,4],name="gt_affine");
        self.gt_offset = tf.placeholder(tf.float32,shape=[None,2,1],name="gt_offset");
        self.affine_pre = tf.square( self.affine - self.gt_affine );
        self.offset_pre = tf.square( self.offset - self.gt_offset );
        self.pre_loss = tf.reduce_mean(self.affine_pre) + tf.reduce_mean(self.offset_pre) + 100.0*self.norm_loss;
    
    def __get_opt__(self):
        self.lr  = tf.placeholder(tf.float32,name='lr');
        self.step = tf.get_variable(shape=[],initializer=tf.constant_initializer(0),trainable=False,name='step',dtype=tf.int32);
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.step);
        self.pre_opt = tf.train.AdamOptimizer(self.lr).minimize(self.pre_loss,global_step=self.step);
        
    def __get_sum__(self):
        sums = [];
        sums.append(tf.summary.scalar("gt_loss",self.gt_loss));
        sums.append(tf.summary.scalar("norm_loss",self.norm_loss));
        sums.append(tf.summary.scalar("reg_loss",self.reg_loss));
        sums.append(tf.summary.scalar("loss",self.loss));
        sums.append(tf.summary.scalar("preloss",self.pre_loss));
        self.sum_op = tf.summary.merge(sums);

if __name__=="__main__":
    with tf.device("/cpu:0"):
        param_shape = [2,22];
        param_init  = tf.constant_initializer([x for x in range(44)],dtype=tf.float32);
        param = tf.get_variable(shape=param_shape,initializer=param_init,trainable=True,name='param');
    ppnet = PPNet("/cpu:0",param);
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        print(sess.run(ppnet.affine));
        
        
