import threading;
try:
    import Queue;
except ImportError:
    import queue as Queue;
import h5py;
import numpy as np;
import os;
import tensorflow as tf;
from scipy.io import loadmat;
from PyQt5 import QtCore, QtGui;
from PyQt5.QtCore import Qt,QCoreApplication;
import sys;
util_path = os.path.dirname(__file__)+os.sep+".."+os.sep+"util";
sys.path.append(util_path);
from QImage2Array import *;

def listdir(dir_,suffix=None):
    lst = os.listdir(dir_);
    olst = [];
    for i in range(len(lst)):
        if suffix is None:
            olst.append( dir_+os.sep+lst[i] );
        elif lst[i].endswith(suffix):
            olst.append( dir_+os.sep+lst[i] );
    return olst;

class Data(threading.Thread):
    def __init__(self,fpath,sizes=[32,256,256,3]):
        super(Data,self).__init__();
        self.sizes = sizes;
        self.fpath = os.path.dirname(fpath);
        self.fname = os.path.basename(fpath).split('.')[0];
        self.datafile = loadmat(fpath); 
        if self.fname=="training":
            self.istrain = True;
            self.namelst = listdir(self.fpath+os.sep+"packed");
            self.namenum = len(self.namelst);
            self.nameidx = int(np.floor(np.random.uniform(0,self.namenum)));
        else:
            self.istrain = False;
            self.namenum = self.datafile[self.fname].shape[1];
            self.nameidx = int(np.floor(np.random.uniform(0,self.namenum)));
        self.Data = Queue.Queue(16);
        self.stopped = False;
        
    def load(self):
        d = {};
        imgs = np.zeros(self.sizes,dtype=np.float32);
        origin_imgs = [];
        whswsh = np.zeros([self.sizes[0],4],dtype=np.int32);
        gt_lbl = [];
        tags = [];
        i = 0;
        while i < self.sizes[0]:
            current_name = self.datafile[self.fname][0,self.nameidx][0][0];
            img = QtGui.QImage(self.fpath+os.sep+"image"+os.sep+"images"+os.sep+current_name+'.jpg');
            need_scale = False;
            if img.height() > 512 or img.width() > 512:
                img = img.scaled(512,512,Qt.KeepAspectRatio);
                need_scale = True;
            if img.isNull():
                self.next_idx();
                continue;
            try:
                lbl = loadmat(self.fpath+'/layout/layout_seg/'+current_name+'.mat')['layout'].copy();
                assert lbl.shape[0] == img.height();
                assert lbl.shape[1] == img.width();
            except:
                self.next_idx();
                continue;
            if need_scale:
                lbl = scaleLabel(lbl,512);
            imgscaled = img.scaled(self.sizes[2],self.sizes[1],Qt.KeepAspectRatio);
            imgpad = QtGui.QImage(self.sizes[2],self.sizes[1],QtGui.QImage.Format_RGB888);
            imgpad.fill(Qt.black);
            painter = QtGui.QPainter();
            painter.begin(imgpad);
            painter.drawImage((self.sizes[2] - imgscaled.width())//2,(self.sizes[1] - imgscaled.height())//2,imgscaled);
            painter.end();
            imgs[i,...] =  convertQImageToArray(imgpad);
            whswsh[i,...] = np.array([img.width(),img.height(),imgscaled.width(),imgscaled.height()],dtype=np.int32);
            tags.append(current_name);
            origin_imgs.append(img);
            gt_lbl.append(lbl);
            self.next_idx();
            i += 1;
        d["img"] = imgs;
        d["origin_img"] = origin_imgs;
        d["gt_lbl"] = gt_lbl;
        d["whswsh"] = whswsh;
        d["tag"] = tags;
        return d;
    
    def next_idx(self):
        self.nameidx += 1;
        if self.nameidx >= self.namenum:
            self.nameidx = 0;
    
    def train_load(self):
        d = {};
        imgs = np.zeros(self.sizes,dtype=np.float32);
        affines = np.zeros([self.sizes[0],3,4],dtype=np.float32);
        offsets = np.zeros([self.sizes[0],2,1],dtype=np.float32);
        gt_xys = np.zeros([self.sizes[0],2,8],dtype=np.float32); 
        tags = [];
        i = 0;
        while i < self.sizes[0]:
            current_name = os.path.basename(self.namelst[self.nameidx]).split(".")[0];
            f = None;
            try:
                f = h5py.File(self.fpath+os.sep+"packed"+os.sep+current_name+".h5","r");
            except:
                self.next_idx();
                continue;
            try:
                imgs[i,...] =  f['img'][...];
                affines[i,...] = f['affine'][...];
                offsets[i,...] = f['offset'][...];
                gt_xys[i,...] = f['layout'][...];
                tags.append(current_name);
            except:
                self.next_idx();
                continue;
            if f is not None:
                f.close();
            self.next_idx();
            i += 1;
        d["img"] = imgs;
        d["affine"] = affines;
        d["offset"] = offsets;
        d["gt_xy"] = gt_xys;
        d["tag"] = tags;
        return d;
        
        
    def run(self):
        while not self.stopped:
            if self.istrain:
                d = self.train_load();
            else:
                d = self.load();
            self.Data.put(d);
        
    def fetch(self):
        if self.stopped:
            return None;
        return self.Data.get();
    
    def shutdown(self):
        self.stopped=True;
        while not self.Data.empty():
            self.Data.get();
        
        
