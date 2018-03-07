# -*- coding: utf-8 -*-
import tensorflow as tf;
import frontnet as fnet;
import sys;
import os;
util_path = os.path.dirname(__file__)+os.sep+".."+os.sep+"util";
sys.path.append(util_path);
from PPNet import PPNet;
from PPL import layout2ResultV2;
from PPL import pixAcc;
from data import Data;
from PyQt5 import QtCore, QtGui;
from PyQt5.QtCore import Qt,QCoreApplication;

def valid(vdata,xyz):
    w = vdata["whswsh"][0,0];
    h = vdata["whswsh"][0,1];
    sw = vdata["whswsh"][0,2];
    sh = vdata["whswsh"][0,3];
    lbl = layout2ResultV2(xyz,256,w,h,sw,sh);
    gt_lbl = vdata["gt_lbl"][0];
    return pixAcc(gt_lbl,lbl);

def train(dev):
    sizes = [32,256,256];
    net = fnet.getNet(net_name,dev,sizes);
    ppnet = PPNet(dev,net["out"]);
    valid_v = tf.placeholder(tf.float32,name='valid_loss');
    valid_sum = tf.summary.scalar("valid_loss",valid_v);
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    saver = tf.train.Saver();
    lrate = 1e-5;
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        train_writer = tf.summary.FileWriter("%s/train"%(dumpdir),graph=sess.graph);
        valid_writer = tf.summary.FileWriter("%s/valid"%(dumpdir),graph=sess.graph);
        ckpt = tf.train.get_checkpoint_state('%s/'%dumpdir);
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess,ckpt.model_checkpoint_path);
        train_data = Data(datadir+os.sep+"training.mat",[32,256,256,3]);
        valid_data = Data(datadir+os.sep+"validation.mat",[1,256,256,3]);
        #pretrain
        try:
            train_data.start();
            valid_data.start();
            for i in range(8):
                data = train_data.fetch();
                _,sum_r,step = sess.run([ppnet.pre_opt,ppnet.sum_op,ppnet.step],
                     feed_dict={
                             net["img"]:data["img"],
                             ppnet.gt_affine:data["affine"],
                             ppnet.gt_offset:data["offset"],
                             ppnet.gt_xy:data["gt_xy"],
                             ppnet.homo_const:ppnet.get_homo_const(sizes[0]),
                             ppnet.homo_box:ppnet.get_homo_box(sizes[0]),
                             ppnet.perspect_mat:ppnet.get_perspect_mat(sizes[0]),
                             ppnet.lr:lrate
                             }
                     );
                train_writer.add_summary(sum_r,step);
                print(step);
            #train 
            for i in range(4096):
                data = train_data.fetch();
                _,sum_r,step = sess.run([ppnet.opt,ppnet.sum_op,ppnet.step],
                    feed_dict={
                             net["img"]:data["img"],
                             ppnet.gt_affine:data["affine"],
                             ppnet.gt_offset:data["offset"],
                             ppnet.gt_xy:data["gt_xy"],
                             ppnet.homo_const:ppnet.get_homo_const(sizes[0]),
                             ppnet.homo_box:ppnet.get_homo_box(sizes[0]),
                             ppnet.perspect_mat:ppnet.get_perspect_mat(sizes[0]),
                             ppnet.lr:lrate
                            }
                     );
                train_writer.add_summary(sum_r,step);
                if i % (train_data.namenum//train_data.sizes[0]) == 0:
                    vdata = valid_data.fetch();
                    xyz1 = sess.run(ppnet.out_hard_with_offset,
                        feed_dict={
                             net["img"]:vdata["img"],
                             ppnet.homo_const:ppnet.get_homo_const(1),
                             ppnet.homo_box:ppnet.get_homo_box(1),
                             ppnet.perspect_mat:ppnet.get_perspect_mat(1),
                            }
                        );
                    xyz = xyz1[0,0:3,:];
                    valid_loss = valid(vdata,xyz);
                    sum_r = sess.run(valid_sum,feed_dict={valid_v:valid_loss});
                    valid_writer.add_summary(sum_r,step);
                    print(valid_loss);
                    saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%(i//(train_data.namenum//train_data.sizes[0])));
                print(step);
        except:
            train_data.shutdown();
            valid_data.shutdown();
        train_data.shutdown();
        valid_data.shutdown();

if __name__=="__main__":
    qapp = QCoreApplication(sys.argv);
    if len(sys.argv) > 1:
        cmd = sys.argv[1];
    else:
        cmd = "train";
    if len(sys.argv) > 2:
        net_name = sys.argv[2];
    else:
        net_name = "PlainCNN";
    if len(sys.argv) > 3: 
        dumpdir = sys.argv[3];
    else:
        dumpdir = os.path.dirname(__file__)+os.sep+".."+os.sep+".."+os.sep+"model";
    if len(sys.argv) > 4:
        datadir = sys.argv[4];
    else:
        datadir = "E:\\WorkSpace\\LSUN";
    if cmd == "train":
            train("/cpu:0");
    else:
        print("Unimplemented cmd");
    
    
