import tensorflow as tf
import numpy as np
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# a = tf.nn.relu(tf.matmul(x, w1))
# y = tf.nn.relu(tf.matmul(a, w2))

# cost function and bp
# clip_by_value 是把数据限定在（1e-10,1.0）之间，避免计算出错
# reduce_mean 求了所有样例的交叉熵的平均值，
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()

    sess.run(init_op)

    print(X.shape)
    print(np.shape(Y))

    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        #every time select batch_size samples to train
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size-1)
        # print(i,start,end)
        # xx = X[start:end]
        # yy = Y[start:end]
        # print("xx shape:%d",xx.shape)
        # print(xx)
        # print("yy shape:%d" , np.shape(yy))
        # print(yy)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        # print(i % 1000)
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("after %d training steps,cross entropy on all data is %g",i,total_cross_entropy)

    print("after training,w1:")
    print(sess.run(w1))
    print("after training,w2:")
    print(sess.run(w2))
