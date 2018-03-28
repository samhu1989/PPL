from PPNet import PPNet;
import tensorflow as tf;
import numpy as np;
class PPCNet(PPNet):
    def __init__(self,dev,param):
        super(PPCNet,self).__init__(dev,param);
        self.input_gt_idx = None;
        self.input_gt_itw = None;
        self.input_gt_w = None;
        self.input_gt_p = None;
        self.feed = None;
        self.batch_size = None;
        self.fv = np.array(
                [
                [4,5,6,7],
                [0,4,5,1],
                [1,5,6,2],
                [2,6,7,3],
                [3,7,4,0]
                ],
                dtype=np.int32);
        
    def __get_loss__(self):
        ###################
        #self.out_xy [None,2,8]
        self.out_xy_t = tf.transpose(self.out_xy,[0,2,1]);
        #[None,8,2];
        self.gt_idx = tf.placeholder(tf.int32,shape=[None,12,2,2],name='gt_idx');
        self.gt_interp_w = tf.placeholder(tf.float32,shape=[None,12,2,1],name='gt_interp_w');
        self.interp_p = tf.gather_nd(self.out_xy_t,self.gt_idx,name='interp_p');
        self.interp_p *= self.gt_interp_w;
        self.interp_p = tf.reduce_sum(self.interp_p,axis=2,name='interp_p');
        self.gt_w = tf.placeholder(tf.float32,shape=[None,12,1],name='gt_w');
        self.gt_p = tf.placeholder(tf.float32,shape=[None,12,2],name='gt_p');
        ###################
        self.gt_dist = self.gt_w*tf.reduce_sum(tf.square(self.interp_p - self.gt_p),axis=1);
        self.gt_loss = tf.reduce_mean(self.gt_dist,name="gt_loss");
        self.out_norm_idx = tf.constant([3],dtype=tf.int32,shape=[1],name="out_norm_idx");
        self.out_norm = tf.gather(self.out,self.out_norm_idx,name="out_norm",axis=1);
        self.norm_loss = tf.reduce_mean(tf.square(self.out_norm - 1.0),name="norm_loss");
        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES);
        if reg:
            self.reg_loss = tf.add_n(self.reg)*0.1;
            self.loss = 10.0*self.gt_loss + self.reg_loss + 1000.0*self.norm_loss ;
        else:
            self.reg_loss = None;
            self.loss = 10.0*self.gt_loss + 1000.0*self.norm_loss;
        return;
        
    def typengt(self,bi,t,pts):
        return eval("self.type%dgt"%t)(bi,pts);
    
    def typenc(self,bi,t,vL):
        return eval("self.type%dc"%t)(bi,vL);
    
    def type4gt(self,bi,pts):
        gt_idx = np.array(
                [
                [[bi,6],[bi,6]],
                [[bi,6],[bi,7]],
                [[bi,6],[bi,5]],
                [[bi,6],[bi,2]],
                [[bi,0],[bi,0]],
                [[bi,0],[bi,0]],
                [[bi,0],[bi,0]],
                [[bi,0],[bi,0]]
                ]
                ,dtype=np.int32);
        gt_itw = np.array(
                [
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]]
                ]
                ,dtype=np.float32);
        gt_w = np.array([[1.0],[1.0],[1.0],[1.0],[0.0],[0.0],[0.0],[0.0]],dtype=np.float32);
        gt_p = np.zeros([8,2],dtype=np.float32);
        gt_p[0:4,:] = pts;
        return gt_idx,gt_itw,gt_w,gt_p;
    
    def type5gt(self,bi,pts):
        gt_idx = np.array(
                [
                [[bi,5],[bi,5]],
                [[bi,4],[bi,5]],
                [[bi,5],[bi,1]],
                [[bi,6],[bi,6]],
                [[bi,7],[bi,6]],
                [[bi,6],[bi,2]],
                [[bi,0],[bi,0]],
                [[bi,0],[bi,0]]
                ]
                ,dtype=np.int32);
        gt_itw = np.array(
                [
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]],
                [[0.5],[0.5]]
                ]
                ,dtype=np.float32);
        gt_w = np.array([[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[0.0],[0.0]],dtype=np.float32);
        gt_p = np.zeros([8,2],dtype=np.float32);
        gt_p[0:6,:] = pts;
        return gt_idx,gt_itw,gt_w,gt_p;
    
    def type4c(self,bi,vL):
        fL = np.zeros([4],dtype=np.int32);
        fL[0] = 0;
        fL[1] = 2;
        if vL[2] == vL[1]:
            fL[2] = 2;
        elif vL[2] == vL[0]:
            fL[2] = 0;
        else:
            fL[2] = 3;
        if vL[3] == vL[0]:
            fL[3] = 0;
        elif vL[3] == vL[1]:
            fL[3] = 2;
        else:
            fL[3] = 3;
        return fL;
    
    def type5c(self,bi,vL):
        fL = np.zeros([4],dtype=np.int32);
        fL = np.array([1,2,3,3],dtype=np.int32);
        return fL;
        
    def init_gt(self,batch_size,lytTp,lytPts,vCorner,vLabel):
        self.input_gt_idx = np.zeros([batch_size,12,2,2],dtype=np.int32);
        self.input_gt_itw = np.zeros([batch_size,12,2,1],dtype=np.float32);
        self.input_gt_w = np.zeros([batch_size,12,1],dtype=np.float32);
        self.input_gt_p = np.zeros([batch_size,12,2],dtype=np.float32);
        self.fL = np.zeros([batch_size,4],dtype=np.int32);
        self.vL = vLabel;
        #gt corners
        for idx,t in enumerate(lytTp):
            gt_idx,gt_itw,gt_w,gt_p = self.typengt(idx,t,lytPts[idx]);
            self.input_gt_idx[idx,0:8,:,:] = gt_idx;
            self.input_gt_itw[idx,0:8,:,:] = gt_itw;
            self.input_gt_w[idx,0:8,:] = gt_w;
            self.input_gt_p[idx,0:8,:] = gt_p;
        #view corners
        self.input_gt_w[:,8:,:] = 1.0;
        self.input_gt_itw[:,8:,:] = 0.5;
        self.input_gt_p[:,8,0] =  vCorner[:,0];
        self.input_gt_p[:,8,1] =  vCorner[:,1];
        self.input_gt_p[:,9,0] = -vCorner[:,0];
        self.input_gt_p[:,9,1] =  vCorner[:,1];
        self.input_gt_p[:,10,0] = -vCorner[:,0];
        self.input_gt_p[:,10,1] = -vCorner[:,1];
        self.input_gt_p[:,11,0] =  vCorner[:,0];
        self.input_gt_p[:,11,1] = -vCorner[:,1];
        for idx,t in enumerate(lytTp):
            self.fL[idx,:] = self.typenc(idx,t,self.vL[idx,:]);
        return;
        
    def get_interp_w(self,p0,p1,pt):
        w = np.zeros([2],dtype=np.float32);
        p0pt = pt - p0;
        p0p1 = p1 - p0;
        n1 = np.sqrt( p0pt[0]*p0pt[0] + p0pt[1]*p0pt[1] );
        n2 = np.sqrt( p0p1[0]*p0p1[0] + p0p1[1]*p0p1[1] );
        if n2 == 0:
            w[0] = 0.5;
            w[1] = 0.5;
            return w;
        proj = ( p0pt[0]*p0p1[0] + p0pt[1]*p0p1[1] ) / n2;
        if proj < 0:
            w[0] = 1.0;
            w[1] = 0.0;
        elif proj > n2:
            w[0] = 0.0;
            w[1] = 1.0;
        else:
            w[1] = proj / n2 ;
            w[0] = 1.0 - w[1];
        return w;
    
    def get_interp_dw(self,p0,p1,pt):
        w = np.zeros([2],dtype=np.float32);
        p0pt = pt - p0;
        p0p1 = p1 - p0;
        p1pt = pt - p1;
        n1 = np.sqrt( p0pt[0]*p0pt[0] + p0pt[1]*p0pt[1] );
        n2 = np.sqrt( p0p1[0]*p0p1[0] + p0p1[1]*p0p1[1] );
        n3 = np.sqrt( p1pt[0]*p1pt[0] + p1pt[1]*p1pt[1] );
        if n2 == 0:
            w[0] = 0.5;
            w[1] = 0.5;
            return n1,w;
        proj = ( p0pt[0]*p0p1[0] + p0pt[1]*p0p1[1] ) / n2;
        if proj < 0:
            w[0] = 1.0;
            w[1] = 0.0;
            d = n1;
        elif proj > n2:
            w[0] = 0.0;
            w[1] = 1.0;
            d = n3;
        else:
            w[1] = proj / n2 ;
            w[0] = 1.0 - w[1];
            d = np.sqrt(n1*n1 - proj*proj);
        return d,w;
    
    def get_interp_c(self,bi,xy,vidx,pt):
        gt_idx = np.zeros([2,2],dtype=np.int32);
        gt_idx[:,0] = bi; 
        gt_itw = np.zeros([2,1],dtype=np.float32);
        gt_itw[:,:] = 0.5;
        mind = 1e5;
        for i in range(4):
            vi0 = vidx[i];
            if i + 1 < 4:
                vi1 = vidx[i+1];
            else:
                vi1 = vidx[0];
            p0 = xy[:,vi0];
            p1 = xy[:,vi1];
            d,w = self.get_interp_dw(p0,p1,pt);
            if d < mind:
                mind = d;
                gt_itw[:,0] = w;
                gt_idx[0,1] = vi0;
                gt_idx[1,1] = vi1;
        return gt_idx,gt_itw;
    
    def update_gt(self,xy):
        #for 1-8 pts update interp_w
        for bi in range(self.batch_size):
            for pi in range(8):
                pi0 = self.input_gt_idx[bi,pi,0,1];
                pi1 = self.input_gt_idx[bi,pi,1,1];
                if pi0 != pi1:
                    p0 = xy[bi,:,pi0];
                    p1 = xy[bi,:,pi1];
                    pt = self.input_gt_p[bi,pi,:];
                    self.input_gt_itw[bi,pi,:,0] = self.get_interp_w(p0,p1,pt);
        for bi in range(self.batch_size):
            for pi in range(4):
                vidx = self.fv[self.fL[bi,pi],:];
                pt = self.input_gt_p[bi,pi+8,:];
                gt_idx,gt_itw = self.get_interp_c(bi,xy[bi,...],vidx,pt);
                self.input_gt_idx[bi,pi+8,:,:] = gt_idx;
                self.input_gt_itw[bi,pi+8,:,:] = gt_itw;
        return;
    
    def get_feed(self,batch_size):
        if (self.feed is None) or (self.batch_size is None) or ( self.batch_size != batch_size ):
            self.batch_size = batch_size;
            self.feed = {
                self.perspect_mat:self.get_perspect_mat(batch_size),
                self.homo_const:self.get_homo_const(batch_size),
                self.homo_box:self.get_homo_box(batch_size),
                self.perspect_mat:self.get_perspect_mat(batch_size),
                self.ext_const:self.get_ext_const(batch_size)
                };
        if self.input_gt_idx is not None:
            self.feed[self.gt_idx] = self.input_gt_idx;
        if self.input_gt_itw is not None:
            self.feed[self.gt_interp_w] = self.input_gt_itw;
        if self.input_gt_w is not None:
            self.feed[self.gt_w] = self.input_gt_w;
        if self.input_gt_p is not None:
            self.feed[self.gt_p] = self.input_gt_p;
        return self.feed;
        
    def __get_opt__(self):
        self.lr  = tf.placeholder(tf.float32,name='lr');
        self.step = tf.get_variable(shape=[],initializer=tf.constant_initializer(0),trainable=False,name='step',dtype=tf.int32);
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.step);
        return;
        
    def __get_sum__(self):
        sums = [];
        sums.append(tf.summary.scalar("gt_loss",self.gt_loss));
        sums.append(tf.summary.scalar("norm_loss",self.norm_loss));
        if self.reg_loss is not None:
            sums.append(tf.summary.scalar("reg_loss",self.reg_loss));
        sums.append(tf.summary.scalar("loss",self.loss));
        self.sum_op = tf.summary.merge(sums);
        
if __name__=="__main__":
    with tf.device("/cpu:0"):
        param_shape = [2,22];
        param_init  = tf.constant_initializer([x for x in range(44)],dtype=tf.float32);
        param = tf.get_variable(shape=param_shape,initializer=param_init,trainable=True,name='param');
    ppnet = PPCNet("/cpu:0",param);
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        print(sess.run(ppnet.affine));
        