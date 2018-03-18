# -*- coding: utf-8 -*-
import numpy as np;
from PPL import pixAcc;
from scipy.io import loadmat;
from scipy.io import savemat;
import os;
from PyQt5 import QtCore, QtGui;
from PyQt5.QtCore import Qt;
from QImage2Array import convertQImageToLabel;
import h5py;
from PPL import layout2ResultV3;
from QImage2Array import convertLabelToQImage;
import sys;

if __name__ == "__main__":
    path = "E:\\WorkSpace\\PPL\\data\\LSUN"
    try:
        train_data = loadmat(path+os.sep+'training.mat');
    except:
        path = '/data4T1/samhu/LSUN'
        train_data = loadmat(path+os.sep+'training.mat');
    res = [];
    score = [];
    for i in range(train_data['training'].shape[1]):
    #for i in range(5):
        name = train_data['training'][0,i][0][0]
        lmat = loadmat(path+os.sep+'layout'+os.sep+'layout_seg'+os.sep+name+'.mat');
        lgt = lmat['layout'].copy();
        img = convertLabelToQImage(lgt);
        imgs = img.scaled(256,256,Qt.KeepAspectRatio);
        if os.path.exists(path+os.sep+'packed'+os.sep+name+'.h5'):
            markf = h5py.File(path+os.sep+'packed'+os.sep+name+'.h5','r');
            xyz = np.zeros([3,8],dtype=np.float32)
            xyz[0:2,:] = markf['layout'];
            lgen = layout2ResultV3(xyz,256,lgt.shape[1],lgt.shape[0],imgs.width(),imgs.height());
            markf.close();
            acc = max(pixAcc(lgt,lgen),pixAcc(lgen,lgt));
            score.append(acc);
            res.append(name);
            print(i,res[-1],score[-1]);
            sys.stdout.flush();
    scorearray = np.array(score);
    resarray = np.array(res);
    scorei = np.argsort(-scorearray);
    s = scorearray[scorei];
    print(s);
    r = resarray[scorei];
    f = open(path+os.sep+"traning_ordered.txt","w")
    for i in range(len(s)):
        f.write("%s,%lf\n"%(r[i],s[i]));
    f.close();

