import tensorflow as tf;
import tflearn as tfl;
def getNet(net_name,dev,sizes):
    return eval(net_name)(dev,sizes);

def PlainCNN(dev,sizes):
    net = {};
    batch_size = sizes[0];
    h = sizes[1];
    w = sizes[2];
    img = tf.placeholder(dtype=tf.float32,shape=[batch_size,h,w,3],name="img");
    net["img"] = img;
    x = tfl.layers.conv.conv_2d(img,16,(7,7),strides=1,activation='relu',weight_decay=1e-5,regularizer='L1');
    net["conv00"] = x;
    x = tfl.layers.conv.conv_2d(  x,16,(7,7),strides=1,activation='relu',weight_decay=1e-5,regularizer='L1');
    net["conv01"] = x;
    x = tfl.layers.conv.conv_2d(  x,32,(7,7),strides=4,activation='relu',weight_decay=1e-5,regularizer='L1');
    net["conv02"] = x;
    i = 3;
    ch = 32;
    while x.shape[1] > 2 and x.shape[2] > 2:
        x = tfl.layers.conv.conv_2d(x,ch,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L1');
        net["conv%02d"%i] = x;
        i += 1;
        x = tfl.layers.conv.conv_2d(x,ch,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L1');
        net["conv%02d"%i] = x;
        i += 1;
        x = tfl.layers.conv.conv_2d(x,2*ch,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L1');
        net["conv%02d"%i] = x;
        i += 1;
        if ch < 512 :
            ch *= 2;
    x = tfl.layers.core.fully_connected(x,128,activation='relu',weight_decay=1e-4,regularizer='L2');
    net["fc%02d"%i] = x;
    i += 1;
    x = tfl.layers.core.fully_connected(x,22,activation='linear',weight_decay=1e-4,regularizer='L2');
    net["fc%02d"%i] = x;
    net["out"] = x;
    i += 1;
    return net;