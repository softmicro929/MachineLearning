#五层神经网络带L2正则化损失函数

import tensorflow as tf
import numpy as np
from numpy.random import RandomState

def get_weight(shape, mlambda):
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(scale=0.1)(var))
    return var



batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

layer_dimension = [2,10,10,10,1]

n_layers = len(layer_dimension)

#当前层
cur_layer = x
#当前层节点个数
in_dimension = layer_dimension[0]

for i in range(1,n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))

    cur_layer = tf.nn.relu((tf.matmul(cur_layer,weight))+bias)
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_-cur_layer))

tf.add_to_collection("losses",mse_loss)

loss = tf.add_n(tf.get_collection("losses"))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# a = tf.nn.relu(tf.matmul(x, w1))
# y = tf.nn.relu(tf.matmul(a, w2))

# cost function and bp
# clip_by_value 是把数据限定在（1e-10,1.0）之间，避免计算出错
# reduce_mean 求了所有样例的交叉熵的平均值，

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1,global_step,16,0.96,staircase=True)
#learning_rate = 0.003
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

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
        sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})
        # print(i % 1000)
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("after %d training steps,cross entropy on all data is %g",i,total_cross_entropy)

    print("after training,w1:")
    print(sess.run(w1))
    print("after training,w2:")
    print(sess.run(w2))
