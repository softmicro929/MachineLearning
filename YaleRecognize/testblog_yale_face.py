#coding:utf-8
"""
python 3
tensorflow 1.1
matplotlib 2.02
"""
import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.0001
batch_size=5

#加载数据
def read_data(filename):
    with open(filename,'rb') as f:
        #记载matlab文件
        dict = sio.loadmat(f)
    return dict['fea'],dict['gnd']

train_data,train_labels = read_data('data/Yale_64x64.mat')
#将标签转为0-14
train_labels = train_labels-1

#shuffle data
np.random.seed(100)
train_data = np.random.permutation(train_data)
np.random.seed(100)
train_labels = np.random.permutation(train_labels)
test_data = train_data[0:50,:]
test_labels = train_labels[0:50]
np.random.seed(200)
test_data = np.random.permutation(test_data)
np.random.seed(200)
test_labels = np.random.permutation(test_labels)

#将标签转为one_hot类型
def label_to_one_hot(labels_dense, num_classes=15):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

#将图片转为灰度图
def to4d(img):
    return img.reshape(img.shape[0],64,64,1).astype(np.float32)/255

train_data = to4d(train_data)
train_labels = label_to_one_hot(train_labels,15)
test_data = to4d(test_data)
test_labels = label_to_one_hot(test_labels,15)

xs = tf.placeholder(tf.float32,[None,64,64,1])
ys = tf.placeholder(tf.float32,[None,15])
keep_prob = tf.placeholder(tf.float32)


#开始构建卷积神经网络
conv1 = tf.layers.conv2d(inputs=xs,filters=32,kernel_size=2,strides=1,padding='same',activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1,pool_size=2,strides=2)
conv2 = tf.layers.conv2d(pool1,filters=72,kernel_size=2,strides=1,padding='same',activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2,pool_size=2,strides=2)
flat = tf.reshape(pool2,[-1,16*16*72])
dense = tf.layers.dense(flat,512,tf.nn.relu)
dropout = tf.nn.dropout(dense,keep_prob)
output = tf.layers.dense(dropout,15)

#计算loss
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys,logits=output)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#返回两个参数一个train_opt,一个acc
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys,axis=1),predictions=tf.argmax(output,axis=1))[1]

with tf.Session() as sess:
    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init)
    for step in range(150):
        i = 0
        while i < len(train_data):
            start = i
            end = i+batch_size
            print(np.shape(train_data[start:end]))
            print(type(train_data[start:end]))
            batch_x = np.array(train_data[start:end])
            print(np.shape(batch_x))
            print(type(batch_x))
            batch_y = np.array(train_labels[start:end])
            _,c = sess.run([train,loss],feed_dict={xs:batch_x,ys:batch_y,keep_prob:0.75})
            i += batch_size
        if step % 1 ==0:
            acc = sess.run(accuracy,feed_dict={xs:test_data,ys:test_labels,keep_prob:1})
            print('= = = = = = > > > > > > ','step:',step,'loss: %.4f'%c,'accuracy: %.2f' %acc)