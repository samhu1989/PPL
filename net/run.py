# -*- coding: utf-8 -*-
import tensorflow as tf;
import frontnet as fnet;
import sys;
import os;
util_path = os.path.dirname(__file__)+os.sep+".."+os.sep+"util";
sys.path.append(util_path);
from PPNet import PPNet;
from PPL import layout2ResultV3;
from PPL import pixAcc;
from data import Data;
from PyQt5 import QtCore, QtGui;
from PyQt5.QtCore import Qt,QCoreApplication;
from QImage2Array import convertArrayToQImage;
from QImage2Array import convertLabelToQImage;

def save_valid(tag,img,gt_lbl,res_lbl):
    if not os.path.exists(dumpdir+os.sep+"valid_pre"):
        os.mkdir(dumpdir+os.sep+"valid_pre");
    fpath = dumpdir+os.sep+"valid_pre"+os.sep+tag+".png";
    gt_img = convertLabelToQImage(gt_lbl);
    res_img = convertLabelToQImage(res_lbl);
    imgpad = QtGui.QImage(img.width()*3+10,img.height(),QtGui.QImage.Format_RGB888);
    imgpad.fill(Qt.white);
    painter = QtGui.QPainter();
    painter.begin(imgpad);
    painter.drawImage(0,0,img);
    painter.drawImage(img.width()+5,0,gt_img);
    painter.drawImage(2*img.width()+10,0,res_img);
    painter.end();
    imgpad.save(fpath);
        
def valid(valid_data,sess,ppnet,net):
    vdata = valid_data.fetch();
    xyz1 = sess.run(ppnet.out_hard_with_offset,
                    feed_dict={
                        net["img"]:vdata["img"],
                        ppnet.homo_const:ppnet.get_homo_const(1),
                        ppnet.homo_box:ppnet.get_homo_box(1),
                        ppnet.perspect_mat:ppnet.get_perspect_mat(1),
                        ppnet.ext_const:ppnet.get_ext_const(1)
                    }
                   );
    xyz = xyz1[0,0:3,:];
    w = vdata["whswsh"][0,0];
    h = vdata["whswsh"][0,1];
    sw = vdata["whswsh"][0,2];
    sh = vdata["whswsh"][0,3];
    lbl = layout2ResultV3(xyz,256,w,h,sw,sh);
    gt_lbl = vdata["gt_lbl"][0];
    valid_loss = max( pixAcc(gt_lbl,lbl) , pixAcc(lbl,gt_lbl) );
    save_valid(vdata["tag"][0],vdata["origin_img"][0],gt_lbl,lbl)
    return valid_loss;

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
        doPreTrain = True;
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess,ckpt.model_checkpoint_path);
            doPreTrain = False;
        train_data = Data(datadir+os.sep+"training.mat",[32,256,256,3]);
        valid_data = Data(datadir+os.sep+"validation.mat",[1,256,256,3]);
        #pretrain
        try:
            train_data.start();
            valid_data.start();
            if doPreTrain:
                for i in range(4096):
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
                             ppnet.ext_const:ppnet.get_ext_const(sizes[0]),
                             ppnet.lr:lrate
                             }
                         );
                    train_writer.add_summary(sum_r,step);
                    if step % (train_data.namenum//train_data.sizes[0]) == 0:
                        valid_loss = valid(valid_data,sess,ppnet,net);
                        print(valid_loss);
                        sum_r = sess.run(valid_sum,feed_dict={valid_v:valid_loss});
                        valid_writer.add_summary(sum_r,step);
                        saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%(step//(train_data.namenum//train_data.sizes[0])));
                    print("pretrain:",step);
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
                             ppnet.ext_const:ppnet.get_ext_const(sizes[0]),
                             ppnet.lr:lrate
                            }
                     );
                train_writer.add_summary(sum_r,step);
                if step % (train_data.namenum//train_data.sizes[0]) == 0:
                    valid_loss = valid(valid_data,sess,ppnet,net);
                    print(valid_loss);
                    sum_r = sess.run(valid_sum,feed_dict={valid_v:valid_loss});
                    valid_writer.add_summary(sum_r,step);
                    saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%(step//(train_data.namenum//train_data.sizes[0])));
                print("train:",step);
        #except:
        finally:
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
    dumpdir += os.sep + net_name;
    if cmd == "train":
            train("/gpu:0");
    else:
        print("Unimplemented cmd");
    
    
