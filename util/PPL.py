from __future__ import print_function;
from __future__ import division;

import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
import os;

class PPBase(object):
    def __init__(self,dev):
        with tf.device( dev ):
            self.__get_box__();
            self.__get_deformed_box__();
            self.__get_perspect_mat__();
            self.__get_project__();
            self.__get_loss__();
            self.__get_opt__();
    def __get_box__(self):
        self.box = tf.constant(
                [ 0.1, 0.1,-0.1,-0.1, 0.1, 0.1,-0.1,-0.1,
                 -0.1, 0.1, 0.1,-0.1,-0.1, 0.1, 0.1,-0.1,
                 -0.2, -0.2, -0.2, -0.2,-0.3,-0.3,-0.3,-0.3],dtype=tf.float32,shape=[3,8],name="box");
    def __get_deformed_box__(self):
        scale_shape = [3,1];
        scale_init  = tf.constant_initializer([1.0,1.0,1.0]);
        self.scale = tf.get_variable(shape=scale_shape,initializer=scale_init ,trainable=True,name='scale');
        t_shape = [3,1];
        t_init  = tf.constant_initializer([0.0,0.0,0.0]);
        self.t = tf.get_variable(shape=t_shape,initializer=t_init ,trainable=True,name='t');
        self.deformed_box = tf.multiply(self.box,self.scale);
        self.deformed_box = tf.add(self.deformed_box,self.t,name="deformed_box");        
            
    def __get_perspect_mat__(self):
        self.fovAngle = np.pi / 4;
        self.fovFar = 1000.0;
        self.fovNear = 0.1;
        self.fovAspect = 1.0;
        cot = 1.0 / np.tan(self.fovAngle/2.0);
        f = self.fovFar;
        n = self.fovNear;
        self.perspect_mat = tf.constant(
            [cot/self.fovAspect,0,0,0,
             0,cot,0,0,
             0,0,-(f+n)/(f-n),-1.0,
             0,0,-2*f*n/(f-n),0
            ],dtype=tf.float32,shape=[4,4],name="perspect_mat");
     
    def __get_project__(self):
        self.homo_box = tf.concat([self.deformed_box,tf.ones(shape=[1,8])],0);
        self.project_box = tf.matmul(self.perspect_mat,self.homo_box,name="project_box");
        norm_factor_shape = [1,8];
        norm_factor_init  = tf.constant_initializer([2.0]);
        self.norm_factor = tf.get_variable(shape=norm_factor_shape,initializer=norm_factor_init ,trainable=True,name='norm_factor');
        self.out = tf.multiply(self.norm_factor,self.project_box,name="out");
        self.out_xy_idx = tf.constant([0,1],dtype=tf.int32,shape=[2],name="out_xy_idx");
        out_xy = tf.gather(self.out,self.out_xy_idx);
        offset_shape = [2,1];
        offset_init  = tf.constant_initializer([0.0,0.0]);
        self.offset = tf.get_variable(shape=offset_shape,initializer=offset_init ,trainable=True,name='offset');
        self.out_xy = tf.add(out_xy,self.offset,name="out_xy");
        w_idx = tf.constant([3],dtype=tf.int32,shape=[1]);
        w = tf.gather(self.project_box,w_idx);
        self.out_hard = self.project_box / w; 
        out_xy_hard = tf.gather(self.out_hard,self.out_xy_idx);
        self.out_xy_hard = tf.add(out_xy_hard,self.offset,name="out_xy_hard");
        
    def __get_loss__(self):
        self.gt_xy = tf.placeholder(tf.float32,shape=[2,8],name='gt_xy');
        self.gt_dist = tf.reduce_sum(tf.square(self.out_xy - self.gt_xy),axis=0);
        self.gt_loss = tf.reduce_mean(self.gt_dist,name="gt_loss");
        self.out_norm_idx = tf.constant([3],dtype=tf.int32,shape=[1],name="out_norm_idx");
        self.out_norm = tf.gather(self.out,self.out_norm_idx,name="out_norm");
        self.norm_loss = tf.reduce_mean(tf.square(self.out_norm - 1.0),name="norm_loss");
        self.loss = self.gt_loss + 100.0*self.norm_loss;
    
    def __get_opt__(self):
        self.lr  = tf.placeholder(tf.float32,name='lr');
        self.step = tf.get_variable(shape=[],initializer=tf.constant_initializer(0),trainable=False,name='step',dtype=tf.int32);
        self.opt = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,global_step=self.step);
        
class PPAffine(PPBase):
    def __init__(self,dev):
        super(PPAffine,self).__init__(dev);
        
    def __get_deformed_box__(self):
        affine_shape = [3,3];
        affine_init  = tf.constant_initializer([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]);
        self.affine = tf.get_variable(shape=affine_shape,initializer=affine_init ,trainable=True,name='affine');
        t_shape = [3,1];
        t_init  = tf.constant_initializer([0.0,0.0,0.0]);
        self.t = tf.get_variable(shape=t_shape,initializer=t_init ,trainable=True,name='t');
        self.deformed_box = tf.matmul(self.affine,self.box);
        self.deformed_box = tf.add(self.deformed_box,self.t,name="deformed_box");
        
class PPWeight(PPAffine):
    def __init__(self,dev):
        with tf.device( dev ):
            self.__get_box__();
            self.__get_deformed_box__();
            self.__get_perspect_mat__();
            self.__get_project__();
            self.__get_interp__();
            self.__get_loss__();
            self.__get_opt__();
      
    def __get_interp__(self):
        self.out_xy_t = tf.transpose(self.out_xy,[1,0]);
        self.interp_v_idx = tf.constant([3,0,0,1,1,2,2,3,7,4,4,5,5,6,6,7,0,4,1,5,2,6,3,7],dtype=tf.int32,shape=[12,2]);
        self.interp_v = tf.gather(self.out_xy_t,self.interp_v_idx,name="interp_v");
        w_shape = [12,2,1];
        w_init  = tf.constant_initializer(
            [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
             0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
            ]);
        self.interp_w = tf.square( tf.get_variable(shape=w_shape,initializer=w_init ,trainable=True,name='interp_w') );
        self.out_interp = tf.transpose(tf.reduce_sum(self.interp_v*self.interp_w,axis=1),[1,0],name='out_interp');
        
    def __get_loss__(self):
        self.gt_xy = tf.placeholder(tf.float32,shape=[2,8],name='gt_xy');
        self.gt_xy_mask = tf.placeholder(tf.float32,shape=[8],name='gt_xy_mask');
        self.xy_dist = tf.reduce_sum(tf.square(self.out_xy - self.gt_xy),axis=0,name="xy_dist");
        self.masked_xy_dist = tf.xy_dist * tf.gt_xy_mask;
        self.xy_loss = tf.reduce_mean(self.masked_xy_dist,name="xy_loss");
        #
        self.gt_interp = tf.placeholder(tf.float32,shape=[2,12],name='gt_interp');
        self.gt_interp_mask = tf.placeholder(tf.float32,shape=[12],name='gt_interp_mask');
        self.interp_dist = tf.reduce_sum(tf.square(self.out_interp - self.gt_interp),axis=0,name="interp_dist");
        self.masked_interp_dist = tf.interp_dist * tf.gt_interp_mask;
        self.interp_loss = tf.reduce_mean(self.masked_xy_dist,name="interp_loss");
        self.gt_loss = self.xy_loss + self.interp_loss;
        #div loss  
        self.out_div_idx = tf.constant([3],dtype=tf.int32,shape=[1],name="out_div_idx");
        self.out_div = tf.gather(self.out,self.out_div_idx,name="out_div");
        self.div_loss = tf.reduce_mean(tf.square(self.out_div - 1.0),name="div_loss");
        #interp loss
        self.interp_loss = tf.reduce_mean( tf.square( tf.reduce_sum(self.interp_w,axis=1) - 1.0 ),name="interp_loss" );
        self.loss = self.gt_loss + 100.0*self.div_loss + 100.0*self.interp_loss;

        
def draw_box2D(size,xy,name=None):
    g1 = [0,1,2,3,0];
    g2 = [4,5,6,7,4];
    g3 = [0,4,1,5,2,6,3,7];
    plt.plot(xy[0,g2],xy[1,g2],'g-');
    for i in range(0,len(g3),2):
        plt.plot(xy[0,g3[i:i+2]],xy[1,g3[i:i+2]],'r-');
    plt.plot(xy[0,g1],xy[1,g1],'b-');
    plt.plot([1,1,-1,-1,1],[1,-1,-1,1,1],'y-')
    plt.show();

def printTensor(tensor,sess):
    print tensor.name,":",sess.run(tensor);
    
def run(ppl,gt):
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    saver = tf.train.Saver();
    lrate = 0.01;
    max_step = 2e5;
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        draw_box2D([320,240],sess.run(ppl.out_xy_hard));
        #print sess.run(ppl.perspect_mat);
        #print sess.run(ppl.out_xy);
        for traincnt in range(int(max_step)):
            _,step = sess.run([ppl.opt,ppl.step],feed_dict={ppl.gt_xy:gt,ppl.lr:lrate});
            if step%(max_step//10)==0:
                lrate *=0.5;
            if step%(max_step//5)==0:
                print sess.run(ppl.loss,feed_dict={ppl.gt_xy:gt});
                #print sess.run(ppl.gt_dist,feed_dict={ppl.gt_xy:gt});
                #printTensor(ppl.norm_factor,sess);
                #printTensor(ppl.norm_loss,sess);
                #printTensor(ppl.affine,sess);
                #printTensor(ppl.scale,sess);
                #printTensor(ppl.t,sess);
                #printTensor(ppl.offset,sess);
                #print sess.run(ppl.out_xy);
        print "hard:"
        draw_box2D([320,240],sess.run(ppl.out_xy_hard));
        print "soft:"
        draw_box2D([320,240],sess.run(ppl.out_xy));
    
def test_run():
    gt = np.array([[ 1.8, 1.8,-1,-1, 1,1,-0.2,-0.2],[1, -1, -1, 1,0.4, -0.4, -0.4, 0.4]],dtype=np.float32);
    draw_box2D([320,240],gt);
    os.environ["CUDA_VISIBLE_DEVICES"] = "0";
    #print "PPBase:"
    #ppl = PPBase("/gpu:0")
    print "PPAffine:"
    ppl = PPAffine("/gpu:0")
    run(ppl,gt);
    return;

def gt_simulate():
    gt_lst = [];
    gt_lst.append({'gt_xy':np.array([[1,1,-1.2,-1.2,0.4,0.4,-0.7,-0.7],[1,-1,-1.2,1.2,0.6,-0.6,-0.8,0.8]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1.44,1,-1.4,-2.16,0.6,0.2,-0.4,-0.9],[1.5,-1,-1,1.5,1,-0.4,-0.4,1]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1,1.15,-1.15,-1,0.9,1,-1,-0.9],[1,-1.2,-1.2,1,0.9,-1,-1,0.9]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1.1,1.1,-1.1,-1.1,0.9,0.9,-1,-1],[1.1,-1.1,-1.4,1.3,0.9,-1,-1.2,1.1]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1.5,1.5,-1.2,-1.2,1,1,0.-0.1,0.-0.1],[1.5,-1.5,-0.95,1.2,1.1,-0.9,-0.7,1]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1.5,1.5,-1.2,-1.2,1,1,0.-0.1,0.-0.1],[1.3,-1.5,-0.95,1.0,0.9,-0.9,-0.7,0.8]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1.8, 1.8,-1.8,-1.8, 1,1,-1,-1],[1, -1, -1, 1,0.4, -0.4, -0.4, 0.4]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1.4, 1.4,-1.2,-1.2, 0.8,0.8,-0.6,-0.6],[1.2, -1.2, -1.2, 1.2,1, -1, -1, 1]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1.2,1.2,-1.1,-1.1,1.0,1.0,-1,-1],[1.0,-1.15,-1.45,1.15,0.75,-1,-1.2,0.9]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1.2,1.3,-1.3,-1.1,1.0,1.1,-1.1,-1],[1.25,-1.0,-1.3,1.4,1.0,-0.7,-0.9,1.15]],dtype=np.float32)});
    gt_lst.append({'gt_xy':np.array([[1.6, 1.6,-1.0,-1.0, 1.0,1.0,-0.4,-0.4],[1.2, -1.2, -1.2, 1.2,1, -1, -1, 1]],dtype=np.float32)});
    return gt_lst;
    

def test_run2():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0";
    print "PPAffine:"
    ppl = PPAffine("/gpu:0");
    gt_lst = gt_simulate();
    cnt = 0;
    for gt in gt_lst:
        print "%d/"%cnt,len(gt_lst);
        draw_box2D([320,240],gt['gt_xy']);
        run(ppl,gt['gt_xy']);
        if cnt > 1:
            break;
        cnt+=1;
        print "%d/"%cnt,len(gt_lst);
    return;
    
def main():
    test_run2();
    
    
if __name__=="__main__":
    main();
    