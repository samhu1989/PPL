import numpy as np;
from PPL import pixAcc;
from scipy.io import loadmat;
from scipy.io import savemat;
import os;
from PyQt5 import QtCore, QtGui;
from PyQt5.QtCore import Qt;
from QImage2Array import convertQImageToLabel;

def listdir(dir_,suffix=None):
    lst = os.listdir(dir_);
    olst = [];
    for i in range(len(lst)):
        if suffix is None:
            olst.append( dir_+os.sep+lst[i] );
        elif lst[i].endswith(suffix):
            olst.append( dir_+os.sep+lst[i] );
    return olst;

class LayoutGroup(object):
    def __init__(self):
        self.mlst = [];
        self.lbl = None;
        
    def tryInsert(self,name,lbl):
        if len(self.mlst)==0 and (self.lbl is None):
            self.mlst.append(name);
            self.lbl = lbl.copy();
            return True;
        if lbl.shape != self.lbl.shape:
            return False;
        elif max(pixAcc(lbl,self.lbl),pixAcc(self.lbl,lbl)) < 0.05:
            self.mlst.append(name);
            return True;
        return False;

if __name__ == "__main__":
    train_data = loadmat("E:\\WorkSpace\\LSUN\\training.mat");
    layoutypeidx = [];
    print(train_data['training'].shape);
    for i in range(train_data['training'].shape[1]):
        idx = train_data['training'][0,i][2][0][0];
        while len(layoutypeidx) <= idx:
            layoutypeidx.append([]);
        layoutypeidx[idx].append(i);
        print(len(layoutypeidx[idx]));
    for i in range(11):
        matdict = train_data.copy();
        idxlst = layoutypeidx[i];
        print(len(idxlst));
        matdict['training'] = train_data['training'][0,idxlst];
        savemat("E:\\WorkSpace\\LSUN\\training%d.mat"%i,matdict);
            
            
        
        
        
        