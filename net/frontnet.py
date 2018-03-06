import tensorflow as tf;
def getNet(net_name,dev,sizes):
    return eval(net_name)(dev,sizes);


def conv2d(imgs,sizes,name):
    

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial,name=)

def PlainCNN(dev,sizes):
    net = {};
    batch_size = sizes[0];
    h = sizes[1];
    w = sizes[2];
    img = tf.placeholder(dtype=tf.float32,shape=[batch_size,h,w,3],name="img");
    net["img"] = img;
    while
    net["out"] = param;
    return net;