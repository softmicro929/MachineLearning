import tensorflow as tf
import numpy as np
import os

from MNIST_CNN import downloadMNIST
from MNIST_CNN import mnist_inference

BATCH_SIZE = 100
# 基础学习率，学习率的衰减率
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

# 正则化项系数，滑动平均衰减率
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99

TRAINING_STEP = 10000

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                                    mnist_inference.NUM_CHANNELS], name="x-input")

    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = mnist_inference.inference(x, None, regularizer)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 将布尔型转为实数型，再计算平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # argmax(),0代表每列中最大，1代表每行中最大
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples,
                                               LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 训练中每过一遍数据，更新一次滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化tf持久化类 saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAINING_STEP):
            xv, yv = mnist.validation.next_batch(BATCH_SIZE)
            reshape_xv = np.reshape(xv,(
                BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            validate_feed = {x: reshape_xv, y_: yv}

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(xs, (
                BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshape_xs, y_: ys})
            # print("after %d training steps,loss on training batch is %g", step, loss_value)
            if i % 1000 == 0:
                print("after %d training steps,loss on training batch is %g" % (step, loss_value))

                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s),validation accuracy using average model is %g " % (step, validate_acc))

                saver.save(sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = downloadMNIST.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
