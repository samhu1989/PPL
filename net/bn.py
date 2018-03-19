import tensorflow as tf;
from tensorflow.python.training.moving_averages import assign_moving_average
def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.shape[-1:];
        moving_mean = tf.get_variable('mean',params_shape,initializer=tf.zeros_initializer,trainable=False);
        moving_variance = tf.get_variable('variance',params_shape,initializer=tf.ones_initializer,trainable=False);
        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, [a for a in range(len(x.shape)-1)] ,name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
    return x;

def instance_norm(x, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='InstatnceNorm2d'):
        params_shape = x.shape[-1:];
        mean, var = tf.nn.moments(x, [a for a in range(1,len(x.shape)-1)] ,keep_dims=True,name='moments');
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, var, None, None, eps)
    return x;