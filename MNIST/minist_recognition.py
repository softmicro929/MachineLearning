import tensorflow as tf

from MNIST import downloadMNIST

INPUT_NODE = 784
OUTPUT_NODE = 10

#配置神经网络参数
LYAER1_NODE = 500
BATCH_SIZE = 100
#基础学习率，学习率的衰减率
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

#正则化项系数，滑动平均衰减率
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99

TRAINING_STEP = 10000

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE],name="y-input")

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LYAER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LYAER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LYAER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    #计算前向传播结果，未使用滑动平均
    y = inference(x,None,weights1,biases1,weights2,biases2)

    global_step = tf.Variable(0,trainable=False)

    #初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算前向传播结果，使用了滑动平均
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

    #argmax(),0代表每列中最大，1代表每行中最大
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

    #设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples, LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    #训练中每过一遍数据，更新一次滑动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    #计算准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #将布尔型转为实数型，再计算平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        #验证数据
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

        # 测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEP):
            if i%1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                test_acc = sess.run(accuracy,feed_dict=test_feed)
                print("after %d training steps,validate_acc is %g", i, validate_acc)
                print("after %d training steps,test_acc is %g", i, test_acc)
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test_acc is %g", TRAINING_STEP, test_acc)

def main(argv=None):
    mnist = downloadMNIST.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()