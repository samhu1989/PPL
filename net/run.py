# -*- coding: utf-8 -*-
import tensorflow as tf;
import frontnet as fnet;
import sys;
import os;
util_path = os.path.dirname(__file__)+os.sep+".."+os.sep+"util";
sys.path.append(util_path);
from PPNet import PPNet;

def train(dev):
    sizes = [32,256,256];
    net = fnet.getNet("PlainCNN",dev,sizes);
    ppnet = PPNet(dev,net["out"]);
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        print(sess.run(ppnet.affine));
        
if __name__=="__main__":
    train("/cpu:0");
    
    
