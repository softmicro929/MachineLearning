import tensorflow as tf

a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')
b = tf.constant([2.0,3.0,4.0],shape=[3],name='b')

c = a+b

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

with tf.device('/gpu:0'):
    print(sess.run(c))