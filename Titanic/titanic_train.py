import tensorflow as tf
import os
import pandas as pd

from Titanic import titanic_data
from Titanic import titanic_inference

BATCH_SIZE = 100
#基础学习率，学习率的衰减率
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

#正则化项系数，滑动平均衰减率
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99

TRAINING_STEP = 20000

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

def train(train_data,test_data):

    # 用正则取出我们要的属性值
    train_df = train_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    # y即Survival结果
    Y = pd.get_dummies(train_df['Survived'], prefix='Survived').as_matrix()

    # X即特征属性值
    X = train_np[:, 1:]

    x = tf.placeholder(tf.float32, [None, titanic_inference.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, titanic_inference.OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = titanic_inference.inference(x, regularizer)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 将布尔型转为实数型，再计算平均值
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # argmax(),0代表每列中最大，1代表每行中最大
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, X.shape[0],LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 训练中每过一遍数据，更新一次滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    merged = tf.summary.merge_all()

    # 初始化tf持久化类 saver
    saver = tf.train.Saver()
    with tf.Session() as sess:

        writer = tf.summary.FileWriter("Log/log", tf.get_default_graph())

        tf.initialize_all_variables().run()

        for i in range(TRAINING_STEP):
            # xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, summary,loss_value, step = sess.run([train_op,merged,loss,global_step],feed_dict={x:X,y_:Y})

            writer.add_summary(summary,i)
            if i % 200 == 0:
                print("after %d training steps,loss on training batch is %g", step, loss_value)
                saver.save(sess,save_path=os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)



    writer.close()



def main(argv=None):
    data_url = "./data"
    train_data,test_data = titanic_data.read_data_sets(data_url)
    train(train_data,test_data)

if __name__ == '__main__':
    tf.app.run()