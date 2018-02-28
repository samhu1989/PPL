from PPL import PPW;
import sys;
import os;
from PyQt5.QtCore import Qt,pyqtSignal,pyqtSlot;
from PyQt5 import QtCore, QtGui, QtOpenGL,QtWidgets;
from PyQt5.QtWidgets import QApplication,QFileDialog,QMainWindow,QMessageBox;
from PyQt5.uic import loadUi;
from scipy.io import loadmat;
from QImage2Array import convertQImageToArray;
from QImage2Array import convertArrayToQImage;
import numpy as np;
import tensorflow as tf;
import h5py;
#This is thread for optimization
class PPThread(QtCore.QThread):
    def __init__(self,parent):
        super().__init__(parent);
        self.timer = QtCore.QTimer();
        self.timer.setSingleShot(False);
        self.timer.moveToThread(self);
        
    def run(self):
        self.timer.start(1);
        self.exec();
#This is work to be done inside the optimization
class PPWork(QtCore.QObject):
    def __init__(self,parent):
        super().__init__(parent);
        self.ppl = PPW("/cpu:0");
        config = tf.ConfigProto();
        config.allow_soft_placement = True;
        self.sess = tf.Session(config=config);
        self.sess.run(tf.global_variables_initializer());
        self.init_affine = np.array(
                [[ 7.68641138,0.0,0.0,0.0],
                 [ 0.0,-7.68641138,0.0,0.0],
                 [ 0.0,0.0,4.77076721,-0.99086928]]
                );
        self.init_offset = np.array(
                [[0.0],[0.0]],dtype=np.float32
                );
        self.init_scale = np.array(
                [[1.0],[1.0]],dtype=np.float32);
        self.sess.run(self.ppl.set_affine,feed_dict={self.ppl.extern_affine:self.init_affine});
        self.lrate = 0.001;
        self.gt_items = None;
        self.gt = np.array([[1.2,1.3,-1.3,-1.1,1.0,1.1,-1.1,-1],[1.25,-1.0,-1.3,1.4,1.0,-0.7,-0.9,1.15]],dtype=np.float32);
        self.w  = np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]],dtype=np.float32);
        self.loss_value = self.sess.run(self.ppl.loss,feed_dict={self.ppl.w:self.w,self.ppl.gt_xy:self.gt});
        
    def reset(self,affine=None,offset=None,scale=None):
        if affine is None:
            self.sess.run(self.ppl.set_affine,feed_dict={self.ppl.extern_affine:self.init_affine});
        else:
            self.sess.run(self.ppl.set_affine,feed_dict={self.ppl.extern_affine:affine});
        if offset is None:
            self.sess.run(self.ppl.set_offset,feed_dict={self.ppl.extern_offset:self.init_offset});
        else:
            self.sess.run(self.ppl.set_offset,feed_dict={self.ppl.extern_offset:offset});
        #if scale is None:
            #self.sess.run(self.ppl.set_scale,feed_dict={self.ppl.extern_scale:self.init_scale});
        #else:
            #self.sess.run(self.ppl.set_scale,feed_dict={self.ppl.extern_scale:scale});
        
    def setPPLWidget(self,w):
        assert isinstance(w,PPLWidget),'Invalid PPLWidget';
        self.pplwidget = w;
     
    def setLearningRate(self,lr):
        self.lrate = lr;
        
    def setTargetItems(self,target):
        self.gt_items = target;
        
    def updateGT(self):
        if self.gt_items is None:
            return;
        else:
            for i,item in enumerate(self.gt_items):
                pos = item.scenePos();
                viewcoord = np.array([[pos.x(),pos.y()]],dtype=np.float32);
                ncoord = self.pplwidget.ViewCoordToNormCoord(viewcoord);
                self.gt[:,i] = ncoord;
        
    def updateW(self):
        if self.gt_items is None:
            return;
        else:
            for i,item in enumerate(self.gt_items):
                self.w[:,i] = item.w;
        
    def getXY(self):
        return self.sess.run(self.ppl.out_xy_hard);
    
    def getAffine(self):
        return self.sess.run(self.ppl.affine);
    
    def getOffset(self):
        return self.sess.run(self.ppl.offset);
    
    #def getScale(self):
        #return self.sess.run(self.ppl.scale);
    
    @pyqtSlot()
    def prepareOptimize(self):
        self.updateGT();
        self.updateW();
    
    @pyqtSlot()
    def optimize(self):
        for i in range(100):
            _,self.loss_value= self.sess.run([self.ppl.opt,self.ppl.loss],feed_dict={self.ppl.w:self.w,self.ppl.gt_xy:self.gt,self.ppl.lr:self.lrate});

class PPTargetItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self,source):
        super().__init__();
        assert isinstance(source,QtWidgets.QGraphicsEllipseItem),"Invalid Source Item"
        self.source = source;
        self.setRect(-6, -6, 12, 12);
        self.setZValue(3.0);
        pen = self.pen();
        pen.setWidth(2);
        pen.setColor(self.source.pen().color());
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255,0)));
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable);
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable);
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable);
        self.setAcceptHoverEvents(True);
        self.setPos(self.source.pos());
        self.initLine();
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges);
        self.w = 1.0;
        self.corners = [];
        self.cp0 = None;
        self.cp1 = None;
        self.withinline = False;
            
    def initLine(self):
        self.line = QtWidgets.QGraphicsLineItem(self);
        pen = self.line.pen();
        pen.setWidth(3);
        pen.setStyle(Qt.DashDotLine);
        pen.setColor(self.source.pen().color());
        self.line.setPen(pen);
        self.updateLine();
        
    def setCorners(self,corners):
        self.corners = corners;
        
    def enforceLineConstraint(self):
        p0p = self.scenePos() - self.cp0;
        p0p1 = self.cp1 - self.cp0;
        p0p1_norm = p0p1.x()*p0p1.x()+p0p1.y()*p0p1.y();
        s = QtCore.QPointF.dotProduct(p0p,p0p1);
        p = s*p0p1/p0p1_norm + self.cp0;
        self.setPos(p);
        self.updateLine();
        
    def setLineConstraint(self,p0,p1):
        self.cp0 = p0;
        self.cp1 = p1;
        self.withinline = True;
        self.enforceLineConstraint();
        
    def unsetLineConstraint(self):
        self.withinline = False;
    
    @pyqtSlot()
    def updateLine(self):
        start = self.line.mapFromScene(self.scenePos());
        end = self.line.mapFromScene(self.source.scenePos());
        self.line.setLine(QtCore.QLineF(start,end));
        
    def getLine(self):
        return self.line;
    
    def mouseReleaseEvent(self,event):
        for item in self.corners:
            fixPos = item.mapFromScene(event.scenePos());
            if item.contains(fixPos):
                self.setPos(item.scenePos());
                self.updateLine();
        if self.withinline:
            self.enforceLineConstraint();
        super().mouseReleaseEvent(event);
    
    def itemChange(self,change,value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            self.updateLine();
        if change == QtWidgets.QGraphicsItem.ItemToolTipHasChanged:
            QtWidgets.QToolTip.showText(self.scenePos().toPoint(),value);
        return super().itemChange(change,value);

            
class PPLWidget(QMainWindow):
    def __init__(self):
        super().__init__()     
        loadUi('./pplui.ui',self);
        self.dataset={};
        self.graphicsScene = QtWidgets.QGraphicsScene();
        self.graphicsView.setScene(self.graphicsScene);
        self.setFocusPolicy(QtCore.Qt.StrongFocus);
        self.origin = QtWidgets.QGraphicsEllipseItem();
        self.origin.setRect(-4, -4, 8, 8);
        self.origin.setZValue(1.0);
        pen = self.origin.pen();
        pen.setWidth(1);
        pen.setColor(QtGui.QColor(255, 0, 0));
        self.origin.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)));
        self.origin.setPen(pen);
        self.viewSize = 256;
        #
        self.imgItem = QtWidgets.QGraphicsPixmapItem();
        self.imgItem.setZValue(-1.0);
        #
        self.mskItem = QtWidgets.QGraphicsPixmapItem();
        self.mskItem.setZValue(0.0);
        self.mskCTable = [
                QtGui.qRgba(216,34,13,180),
                QtGui.qRgba(191,221,163,180),
                QtGui.qRgba(255,233,169,180),
                QtGui.qRgba(240,145,146,180),
                QtGui.qRgba(249,209,212,180),
                ]
        self.CTable = [
                QtGui.qRgb(129,194,214),
                QtGui.qRgb(129,146,214),
                QtGui.qRgb(217,179,230),
                QtGui.qRgb(220,247,161),
                QtGui.qRgb(131,252,216),
                QtGui.qRgb(253,255,223),
                QtGui.qRgb(243,215,181),
                QtGui.qRgb(255,233,87)
                ]
        self.cornerItems = [];
        #
        self.graphicsScene.addItem(self.imgItem);
        self.graphicsScene.addItem(self.mskItem);
        self.graphicsScene.addItem(self.origin);
        #
        self.initWork();
        self.initLayout();
        self.initTarget();
        self.connectWork();
        self.drawBound();
        #
        self.actionSet_LSUN_Root.triggered.connect(self.setLSUNRoot);
        self.index = 0;
        self.LSUNRoot = None;
        self.actionLast.triggered.connect(self.loadLast);
        self.actionNext.triggered.connect(self.loadNext);
        self.actionReset_Layout.triggered.connect(self.resetLayout);
    
    def closeEvent(self,event):
        self.saveCurrent();
        super().closeEvent(event);
    
    def loaddataset(self,fname):
        basename = fname.split('.')[-2];
        try:
            self.dataset[basename] = loadmat(self.LSUNRoot+'/'+fname);
        except:
            QMessageBox.warning(
                    self,
                    "Missing File",  
                    "Failed to open "+self.LSUNRoot + "/" + fname +",Please check the dataset",
                    QMessageBox.Ok
                    );
                                
    @pyqtSlot()
    def setLSUNRoot(self):
        self.LSUNRoot = QFileDialog.getExistingDirectory(self,'Open Dir','E:/WorkSpace/');
        self.loaddataset("training.mat");
        self.loaddataset("validation.mat");
        self.loaddataset("testing.mat");
        self.index = 0;
        self.train_num = self.dataset['training']['training'].size;
        self.valid_num = self.dataset['validation']['validation'].size;
        self.loadCurrent();
        
    @pyqtSlot()
    def resetLayout(self):
        self.work.reset();
        self.updateLayout();
        for item in self.targetItems:
            item.updateLine();
        
    @pyqtSlot()    
    def loadNext(self):
        if self.LSUNRoot is None:
            return;
        if self.index < (self.train_num + self.valid_num - 1):
            self.saveCurrent();
            self.index += 1;
            self.loadCurrent();
        
    @pyqtSlot()
    def loadLast(self):
        if self.LSUNRoot is None:
            return;
        if self.index > 0 :
            self.saveCurrent();
            self.index -= 1;
            self.loadCurrent();
    
    def loadCurrent(self):
        if self.index < self.train_num:
            self.current_name = self.dataset['training']['training'][0,self.index][0][0];
        else:
            self.current_name = self.dataset['validation']['validation'][0,self.index-self.train_num][0][0];
        self.loadPixmap();
        self.loadMask();           
        w = self.img.width();
        h = self.img.height();
        sw = self.imgscaled.width();
        sh = self.imgscaled.height();
        if self.index < self.train_num:
            self.current_corners = self.ImgCoordToNormCoord(w,h,sw,sh,self.dataset['training']['training'][0,self.index][3][:]);
        else:
            self.current_corners = self.ImgCoordToNormCoord(w,h,sw,sh,self.dataset['validation']['validation'][0,self.index-self.train_num][3][:]);
        self.loadCorners();
        for item in self.targetItems:
            item.setCorners(self.cornerItems);
            item.unsetLineConstraint();
        self.loadExist();
            
    def loadExist(self):
        if not os.path.exists(self.LSUNRoot+"/packed"):
            os.mkdir(self.LSUNRoot+"/packed");
        if not os.path.exists(self.LSUNRoot+"/packed/"+self.current_name+".h5"):
            self.packfile = h5py.File(self.LSUNRoot+"/packed/"+self.current_name+".h5",'w');
        else:
            self.packfile = h5py.File(self.LSUNRoot+"/packed/"+self.current_name+".h5",'r+');
        if 'img' in self.packfile.keys():
            img = convertArrayToQImage(self.packfile['img'][...]);
            self.imgItem.setPixmap(QtGui.QPixmap.fromImage(img));
            self.imgItem.setPos(-self.viewSize//2,-self.viewSize//2);
        else:
            self.packfile.create_dataset('img',[256,256,3],dtype='float32',compression="gzip");            
        if ( 'affine' in self.packfile.keys() ) and ( 'offset' in self.packfile.keys() ):
            self.work.reset(
                    self.packfile['affine'][...],
                    self.packfile['offset'][...]);
            self.updateLayout();
            for item in self.targetItems:
                item.updateLine();
            if 'layout' in self.packfile.keys():
                loss = np.mean(np.sum(np.square(self.work.sess.run(self.work.ppl.out_xy_hard) - self.packfile['layout'][...]),axis=0));
                self.statusbar.showMessage("[%d]:reload loss:%f"%(self.index,loss));
        else:
            self.packfile.create_dataset('affine',[3,4],dtype='float32',compression="gzip");
            self.packfile.create_dataset('offset',[2,1],dtype='float32',compression="gzip");
            self.packfile.create_dataset('layout',[2,8],dtype='float32',compression="gzip");
            
    def saveCurrent(self):
        img = self.imgItem.pixmap().toImage();
        self.packfile['img'][...] = convertQImageToArray(img);
        self.packfile['layout'][...] = self.work.getXY();
        self.packfile['affine'][...] = self.work.getAffine();
        self.packfile['offset'][...] = self.work.getOffset();
        self.packfile.close();
        
    def loadPixmap(self):
        self.img = QtGui.QImage(self.LSUNRoot+'/image/images/'+self.current_name+'.jpg');
        self.imgscaled = self.img.scaled(self.viewSize,self.viewSize,Qt.KeepAspectRatio);
        self.imgpad = QtGui.QImage(self.viewSize,self.viewSize,QtGui.QImage.Format_RGB888);
        self.imgpad.fill(Qt.black);
        painter = QtGui.QPainter();
        painter.begin(self.imgpad);
        painter.drawImage((self.viewSize - self.imgscaled.width())//2,(self.viewSize - self.imgscaled.height())//2,self.imgscaled);
        painter.end();
        self.imgItem.setPixmap(QtGui.QPixmap.fromImage(self.imgpad));
        self.imgItem.setPos(-self.viewSize//2.0,-self.viewSize//2.0);
        
    def loadMask(self):
        self.mskmat = loadmat(self.LSUNRoot+'/layout/layout_seg/'+self.current_name+'.mat')['layout'].copy();
        self.msk = QtGui.QImage(self.mskmat,self.mskmat.shape[1],self.mskmat.shape[0],self.mskmat.shape[1],QtGui.QImage.Format_Indexed8);
        self.msk.setColorTable(self.mskCTable);
        self.mskItem.setPixmap(QtGui.QPixmap.fromImage(self.msk).scaled(self.viewSize,self.viewSize,Qt.KeepAspectRatio));
        self.mskItem.setPos(-self.mskItem.pixmap().width()/2.0,-self.mskItem.pixmap().height()/2.0);
        
    def ImgCoordToNormCoord(self,w,h,sw,sh,coord):
        newcoord = coord / np.array([[float(w/sw*self.viewSize),float(h/sh*self.viewSize)]],dtype=np.float32);
        newcoord *= 2.0;
        newcoord -= np.array([[float(sw/self.viewSize),float(sh/self.viewSize)]],dtype=np.float32);
        newcoord[:,1] *= -1.0;
        return newcoord;
    
    def NormCoordToImgCoord(self,w,h,sw,sh,coord):
        newcoord = coord.copy();
        newcoord[:,1] *= -1.0;
        newcoord += np.array([[float(sw/self.viewSize),float(sh/self.viewSize)]],dtype=np.float32);
        newcoord /= 2.0;
        newcoord *= np.array([[float(w/sw*self.viewSize),float(h/sh*self.viewSize)]],dtype=np.float32);
        return newcoord;
    
    def ViewCoordToNormCoord(self,coord):
        newcoord = coord.copy();
        newcoord[:,1] *= -1.0;
        newcoord /= (self.viewSize/2.0);
        return newcoord;
    
    def NormCoordToViewCoord(self,coord):
        newcoord = coord.copy();
        newcoord[:,1] *= -1.0;
        newcoord *= (self.viewSize/2.0);
        return newcoord;
        
    def loadCorners(self):
        #print(self.current_corners);
        for item in self.cornerItems:
            self.graphicsScene.removeItem(item);
        self.cornerItems.clear();
        self.current_view_corners = self.NormCoordToViewCoord(self.current_corners);
        for i in range(self.current_view_corners.shape[0]):
            corner = QtWidgets.QGraphicsEllipseItem();
            corner.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable);
            corner.setRect(-4, -4, 8, 8);
            corner.setZValue(1.0);
            pen = corner.pen();
            pen.setWidth(1);
            pen.setColor(QtGui.QColor(self.CTable[i]));
            corner.setBrush(QtGui.QBrush(QtGui.QColor(255,0,0)));
            corner.setPen(pen);
            self.graphicsScene.addItem(corner);
            x = self.current_view_corners[i,0];
            y = self.current_view_corners[i,1];
            corner.setPos(x,y)
            self.cornerItems.append(corner);
            
    @pyqtSlot()
    def updateLayout(self):
        self.boxCornerCoord = self.work.getXY();
        boxCornerViewCoord = self.NormCoordToViewCoord(np.transpose(self.boxCornerCoord,[1,0]));
        for i, item in enumerate(self.boxCornerItems):
            scenePos = QtCore.QPointF(boxCornerViewCoord[i,0],boxCornerViewCoord[i,1]);
            item.setPos(scenePos);
        lineIdx = [0,1,1,2,2,3,3,0,4,5,5,6,6,7,7,4,0,4,1,5,2,6,3,7];
        boxLineViewCoord = boxCornerViewCoord[lineIdx,:];
        for i, item in enumerate(self.boxLineItems):
            start = QtCore.QPointF(boxLineViewCoord[2*i,0],boxLineViewCoord[2*i,1]);
            end = QtCore.QPointF(boxLineViewCoord[2*i+1,0],boxLineViewCoord[2*i+1,1]);
            item.setLine(QtCore.QLineF(start,end));
            
    @pyqtSlot()
    def layout2Res(self):
        return;
            
    @pyqtSlot()
    def showLoss(self):
        self.statusbar.showMessage("%f"%self.work.loss_value);
            
    def drawBound(self):
        self.boundCornerCoord = np.array([[2,2,-2,-2],[2,-2,-2,2]],dtype=np.float32);
        self.boundLineItems = [];
        boundCornerViewCoord = self.NormCoordToViewCoord(np.transpose(self.boundCornerCoord,[1,0]));
        boundLineIdx = [0,1,1,2,2,3,3,0];
        boundLineViewCoord = boundCornerViewCoord[boundLineIdx,:];
        for i in range(4):
            line = QtWidgets.QGraphicsLineItem();
            pen = line.pen();
            pen.setWidth(3);
            pen.setColor(QtGui.QColor(0, 0, 0));
            line.setPen(pen);
            start = QtCore.QPointF(boundLineViewCoord[2*i,0],boundLineViewCoord[2*i,1]);
            end = QtCore.QPointF(boundLineViewCoord[2*i+1,0],boundLineViewCoord[2*i+1,1]);
            line.setLine(QtCore.QLineF(start,end));
            self.graphicsScene.addItem(line);
            self.boundLineItems.append(line);
            
    def initLayout(self):
        self.boxCornerItems = [];
        for i in range(8):
            corner = QtWidgets.QGraphicsEllipseItem();
            corner.setRect(-4, -4, 8, 8);
            corner.setZValue(2.0);
            pen = corner.pen();
            pen.setWidth(1);
            if i < 4:
                pen.setColor(QtGui.QColor(0, 0, 255));
                corner.setPen(pen);
                corner.setBrush(QtGui.QBrush(QtGui.QColor(0,0,255)));
            else:
                pen.setColor(QtGui.QColor(0, 255, 0));
                corner.setPen(pen);
                corner.setBrush(QtGui.QBrush(QtGui.QColor(0, 255,0)));
            self.graphicsScene.addItem(corner);
            self.boxCornerItems.append(corner);
        self.boxLineItems = [];
        for i in range(12):
            line = QtWidgets.QGraphicsLineItem();
            pen = line.pen();
            pen.setWidth(3);
            if i < 4:
                pen.setColor(QtGui.QColor(0, 0, 255));
            elif i < 8:
                pen.setColor(QtGui.QColor(0, 255, 0));
            else:
                pen.setColor(QtGui.QColor(255, 0, 0));
            line.setPen(pen);
            line.setZValue(2.0-float(i)/12.0);            
            self.graphicsScene.addItem(line);
            self.boxLineItems.append(line);
        self.updateLayout();
            
    def initTarget(self):
        self.targetItems = [];
        for i in range(8):
            corner = PPTargetItem(self.boxCornerItems[i]);
            self.graphicsScene.addItem(corner);
            self.targetItems.append(corner);
        
    def initWork(self):
        self.thread = PPThread(self);
        self.work = PPWork(None);
        self.work.setPPLWidget(self);        

    def connectWork(self):
        self.work.setTargetItems(self.targetItems);
        self.work.moveToThread(self.thread);
        self.thread.timer.timeout.connect(self.work.prepareOptimize);
        self.thread.timer.timeout.connect(self.work.optimize);
        self.thread.timer.timeout.connect(self.updateLayout);
        for item in self.targetItems:
            self.thread.timer.timeout.connect(item.updateLine);
        self.thread.timer.timeout.connect(self.showLoss);
        
    def keyPressEvent(self, event):
        if not type(event) == QtGui.QKeyEvent:
            return;
        if event.key() == QtCore.Qt.Key_Space:
            self.thread.start(QtCore.QThread.HighPriority);
        if event.key() == QtCore.Qt.Key_G and event.modifiers() == QtCore.Qt.ControlModifier:
            items = self.graphicsScene.selectedItems();
            if len(items)==3 :
                titemcnt = 0;
                citemcnt = 0;
                titem = None;
                citem = [];
                for item in items:
                    if item in self.targetItems:
                        titemcnt += 1;
                        titem = item;
                    if item in self.cornerItems:
                        citemcnt += 1;
                        citem.append(item);
                if titemcnt == 1 and citemcnt == 2:
                    titem.setLineConstraint(citem[0].scenePos(),citem[1].scenePos());
            elif len(items)==1:
                if items[0] in self.targetItems:
                    items[0].unsetLineConstraint();
        if event.key() == QtCore.Qt.Key_W:
            items = self.graphicsScene.selectedItems();
            if len(items)==1:
                if items[0] in self.targetItems:
                    if items[0].w < 1.0:
                        items[0].w += 0.005;
                    items[0].setToolTip("%f"%items[0].w);
                    pos = self.graphicsView.mapFromScene(items[0].scenePos());
                    pos = self.graphicsView.mapToParent(pos);
                    pos = self.mapToGlobal(pos);
                    QtWidgets.QToolTip.showText(pos,items[0].toolTip(),self);
        if event.key() == QtCore.Qt.Key_S:
            items = self.graphicsScene.selectedItems();
            if len(items)==1:
                if items[0] in self.targetItems:
                    if items[0].w >= 0.005:
                        items[0].w -= 0.005;
                    items[0].setToolTip("%f"%items[0].w);
                    pos = self.graphicsView.mapFromScene(items[0].scenePos());
                    pos = self.graphicsView.mapToParent(pos);
                    pos = self.mapToGlobal(pos);
                    QtWidgets.QToolTip.showText(pos,items[0].toolTip(),self);
        QtWidgets.QMainWindow.keyPressEvent(self,event);
             
    def keyReleaseEvent(self, event):
        if type(event) == QtGui.QKeyEvent and event.key() == QtCore.Qt.Key_Space:
            self.thread.quit();
        QtWidgets.QMainWindow.keyReleaseEvent(self,event);
        
if __name__ == '__main__':
    app = QApplication(sys.argv);
    w = PPLWidget();
    w.show();
    sys.exit(app.exec_());