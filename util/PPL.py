from __future__ import print_function;
from __future__ import division;

import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
import os;
from PyQt5.QtCore import Qt,pyqtSignal,pyqtSlot;

class PPBase(object):
    def __init__(self,dev):
        with tf.device( dev ):
            self.__get_box__();
            self.__get_deformed_box__();
            self.__get_perspect_mat__();
            self.__get_project__();
            self.__get_viewport_map__();
            self.__get_loss__();
            self.__get_opt__();
            
    def __get_box__(self):
        self.box = tf.constant(
                [ 0.1, 0.1,-0.1,-0.1, 0.1, 0.1,-0.1,-0.1,
                 -0.1, 0.1, 0.1,-0.1,-0.1, 0.1, 0.1,-0.1,
                  0.1, 0.1, 0.1, 0.1,-0.1,-0.1,-0.1,-0.1],dtype=tf.float32,shape=[3,8],name="box");
    def __get_deformed_box__(self):
        scale_shape = [3,1];
        scale_init  = tf.constant_initializer([1.0,1.0,1.0]);
        self.scale = tf.get_variable(shape=scale_shape,initializer=scale_init ,trainable=True,name='scale');
        t_shape = [3,1];
        t_init  = tf.constant_initializer([0.0,0.0,0.0]);
        self.t = tf.get_variable(shape=t_shape,initializer=t_init ,trainable=True,name='t');
        self.deformed_box = tf.multiply(self.box,self.scale);
        self.deformed_box = tf.add(self.deformed_box,self.t,name="deformed_box");
        self.homo_deformed_box = tf.concat([self.deformed_box,tf.ones(shape=[1,8])],0,name="homo_deformed_box");        
            
    def __get_perspect_mat__(self):
        self.fovAngle = np.pi / 2;
        self.fovFar = 1000.0;
        self.fovNear = 0.1;
        self.fovAspect = 1.0;
        self.cot = 1.0 / np.tan(self.fovAngle/2.0);
        f = self.fovFar;
        n = self.fovNear;
        self.perspect_mat = tf.constant(
            [self.cot/self.fovAspect,0,0,0,
             0,self.cot,0,0,
             0,0,-(f+n)/(f-n),-2*f*n/(f-n),
             0,0,-1.0,0
            ],dtype=tf.float32,shape=[4,4],name="perspect_mat");
     
    def __get_project__(self):
        self.project_box = tf.matmul(self.perspect_mat,self.homo_deformed_box,name="project_box");
        norm_factor_shape = [1,8];
        norm_factor_init  = tf.constant_initializer([2.0]);
        self.norm_factor = tf.get_variable(shape=norm_factor_shape,initializer=norm_factor_init ,trainable=True,name='norm_factor');
        self.out = tf.multiply(self.norm_factor,self.project_box,name="out");
        #
        w_idx = tf.constant([3],dtype=tf.int32,shape=[1]);
        w = tf.gather(self.project_box,w_idx);
        self.out_hard = self.project_box / w; 

        
    def __get_viewport_map__(self):
        offset_shape = [2,1];
        offset_init  = tf.constant_initializer([0.0,0.0]);
        self.offset = tf.get_variable(shape=offset_shape,initializer=offset_init ,trainable=True,name='offset');
        self.extern_offset = tf.placeholder(tf.float32,shape=[2,1],name='extern_offset');
        self.set_offset = tf.assign(self.offset,self.extern_offset);
        
        self.out_xy_idx = tf.constant([0,1],dtype=tf.int32,shape=[2],name="out_xy_idx");
        
        out_xy = tf.gather(self.out,self.out_xy_idx);
        self.out_xy = tf.add(out_xy,self.offset,name="out_xy");
        
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
        #self.opt = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,global_step=self.step);
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.step);
        
class PPAffine(PPBase):
    def __init__(self,dev):
        super(PPAffine,self).__init__(dev);
        
    def __get_deformed_box__(self):
        affine_shape = [3,4];
        affine_init  = tf.constant_initializer([1.0,0.0,0.0,0.0,
                                                0.0,1.0,0.0,0.0,
                                                0.0,0.0,1.0,-0.5]);
        self.affine = tf.get_variable(shape=affine_shape,initializer=affine_init ,trainable=True,name='affine');
        self.extern_affine = tf.placeholder(tf.float32,shape=[3,4],name='extern_affine');
        self.set_affine = tf.assign(self.affine,self.extern_affine);
        homo_const = tf.constant([0,0,0,1],dtype=tf.float32,shape=[1,4],name="homo_const");
        self.affine_mat = tf.concat([self.affine,homo_const],0);
        homo_box = tf.concat([self.box,tf.ones(shape=[1,8])],0);
        self.homo_deformed_box = tf.matmul(self.affine_mat,homo_box,name="homo_deformed_box");
        
#with weight        
class PPW(PPAffine):
    def __init__(self,dev):
        super(PPW,self).__init__(dev);
        
    def __get_loss__(self):
        self.w = tf.placeholder(tf.float32,shape=[1,8],name='w');
        self.gt_xy = tf.placeholder(tf.float32,shape=[2,8],name='gt_xy');
        self.gt_dist = tf.reduce_sum(tf.square(self.out_xy - self.gt_xy),axis=0);
        self.gt_loss = tf.reduce_mean(self.w*self.gt_dist,name="gt_loss");
        self.out_norm_idx = tf.constant([3],dtype=tf.int32,shape=[1],name="out_norm_idx");
        self.out_norm = tf.gather(self.out,self.out_norm_idx,name="out_norm");
        self.norm_loss = tf.reduce_mean(tf.square(self.out_norm - 1.0),name="norm_loss");
        self.loss = self.gt_loss + 100.0*self.norm_loss;
        
class PPV(PPW):
    def __init__(self,dev):
        super(PPV,self).__init__(dev);
        
    def __get_perspect_mat__(self):
        self.fovAngle = np.pi / 2;
        self.fovFar = 1000.0;
        self.fovNear = 0.1;
        self.fovAspect = 1.0;
        self.cot = 1.0 / np.tan(self.fovAngle/2.0);
        f = self.fovFar;
        n = self.fovNear;
        self.perspect_const = tf.constant([0.0,-1.0],dtype=tf.float32,shape=[2],name="perspect_const");
        perspect_var_shape = [4];
        perspect_var_init  = tf.constant_initializer([self.cot/self.fovAspect,self.cot,-(f+n)/(f-n),-2*f*n/(f-n)]);
        self.perspect_var = tf.get_variable(shape=perspect_var_shape,initializer=perspect_var_init,trainable=True,name='perspect_var');
        self.extern_perspect_var = tf.placeholder(tf.float32,shape=[4],name='extern_perspect_var');
        self.set_perspect_var = tf.assign(self.perspect_var,self.extern_perspect_var);
        perspect_val = tf.concat([self.perspect_const,self.perspect_var],0);
        perspect_idx = tf.constant(
                [2,0,0,0,
                 0,3,0,0,
                 0,0,4,5,
                 0,0,1,0],dtype=tf.int32,shape=[4,4],name="perspect_idx");
        self.perspect_mat = tf.gather(perspect_val,perspect_idx,name="perspect_mat");
        
class PPS(PPW):
    def __init__(self,dev):
        super(PPS,self).__init__(dev);
        
    def __get_viewport_map__(self):
        offset_shape = [2,1];
        offset_init  = tf.constant_initializer([0.0,0.0]);
        self.offset = tf.get_variable(shape=offset_shape,initializer=offset_init ,trainable=True,name='offset');
        self.extern_offset = tf.placeholder(tf.float32,shape=[2,1],name='extern_offset');
        self.set_offset = tf.assign(self.offset,self.extern_offset);
        
        scale_shape = [2,1];
        scale_init = tf.constant_initializer([1.0,1.0]);
        self.scale = tf.get_variable(shape=scale_shape,initializer=scale_init ,trainable=True,name='scale');
        self.extern_scale = tf.placeholder(tf.float32,shape=[2,1],name='extern_scale');
        self.set_scale = tf.assign(self.scale,self.extern_scale);
        
        self.out_xy_idx = tf.constant([0,1],dtype=tf.int32,shape=[2],name="out_xy_idx");
        
        out_xy = tf.gather(self.out,self.out_xy_idx);
        self.out_xy = tf.multiply(tf.add(out_xy,self.offset),self.scale,name="out_xy");
        
        out_xy_hard = tf.gather(self.out_hard,self.out_xy_idx);
        self.out_xy_hard = tf.multiply(tf.add(out_xy_hard,self.offset),self.scale,name="out_xy_hard");
        
def NormCoordToImgCoord(viewSize,w,h,sw,sh,coord):
    newcoord = coord.copy();
    newcoord[:,1] *= -1.0;
    newcoord += np.array([[float(sw/viewSize),float(sh/viewSize)]],dtype=np.float32);
    newcoord /= 2.0;
    newcoord *= np.array([[float(w/sw*viewSize),float(h/sh*viewSize)]],dtype=np.float32);
    return newcoord;

def areaFromL(a,b,c):
    return 0.25*np.sqrt((a+b+c)*(a+b-c)*(a+c-b)*(b+c-a));

def areaFromV(p0,p1,p2):
    a = np.sqrt((np.square(p1-p0)).sum());
    b = np.sqrt((np.square(p2-p1)).sum());
    c = np.sqrt((np.square(p0-p2)).sum());
    return areaFromL(a,b,c);

def isInside(ABCD,M):
    return;
    
    
    

def layout2Label(xyz,x,y):
    bestL = -1;
    bestZ = 1e9;
    fidx = np.array(
            [[4,5,6,7],
             [0,4,7,3],
             [1,5,4,0],
             [2,6,5,1],
             [3,7,6,2]],dtype=np.int32);
    for i in range(5):
        fxyz = xyz[fidx[i,:],:];        
        if bestZ > z:
            bestZ = z;
            bestL = i;
    return bestL;
        
def layout2Res(xyz,viewSize,img):
    newxyz = xyz.copy();
    xy = newxyz[:,0:2];
    imgscaled = img.scaled(viewSize,viewSize,Qt.KeepAspectRatio);
    w = img.width();
    h = img.height();
    sw = imgscaled.width();
    sh = imgscaled.height();
    newxyz[:,0:2] = NormCoordToImgCoord(viewSize,w,h,sw,sh,xy);
    lbl = np.zeros([h,w],dtype=np.uint8);
    for y in range(h):
        for x in range(w):
            lbl[y,x] = layout2Label(newxyz,x,y);
        
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
    print(tensor.name,":",sess.run(tensor));
    
def run(ppl,gt):
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    lrate = 0.005;
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
                print(sess.run(ppl.loss,feed_dict={ppl.gt_xy:gt}));
                #print sess.run(ppl.gt_dist,feed_dict={ppl.gt_xy:gt});
                #printTensor(ppl.norm_factor,sess);
                #printTensor(ppl.norm_loss,sess);
                printTensor(ppl.affine,sess);
                #printTensor(ppl.scale,sess);
                #printTensor(ppl.t,sess);
                #printTensor(ppl.offset,sess);
                #print sess.run(ppl.out_xy);
        print("hard:");
        draw_box2D([320,240],sess.run(ppl.out_xy_hard));
        print("soft:");
        draw_box2D([320,240],sess.run(ppl.out_xy));
    
def test_run():
    gt = np.array([[ 1.8, 1.8,-1,-1, 1,1,-0.2,-0.2],[1, -1, -1, 1,0.4, -0.4, -0.4, 0.4]],dtype=np.float32);
    draw_box2D([320,240],gt);
    os.environ["CUDA_VISIBLE_DEVICES"] = "0";
    print("PPAffine:");
    ppl = PPAffine("/gpu:0")
    run(ppl,gt);
    return;

def gt_simulate():
    gt_lst = [];
    gt_lst.append({'gt_xy':np.array([[1.5,1.5,-1.5,-1.5,0.5,0.5,-0.5,-0.5],[1.5,-1.5,-1.5,1.5,0.5,-0.5,-0.5,0.5]])});
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
    print("PPAffine:");
    ppl = PPAffine("/gpu:0");
    gt_lst = gt_simulate();
    cnt = 0;
    for gt in gt_lst:
        if cnt >= 5:
            break;
        print("%d/"%cnt,len(gt_lst));
        draw_box2D([320,240],gt['gt_xy']);
        run(ppl,gt['gt_xy']);
        cnt+=1;
        print("%d/"%cnt,len(gt_lst));
    return;
    
def main():
    test_run2();
    
    
if __name__=="__main__":
    main();
    