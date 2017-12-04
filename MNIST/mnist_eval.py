import tensorflow as tf
import time

from MNIST import downloadMNIST
from MNIST import mnist_inference
from MNIST import mnist_train

EVAL_SECOND = 10

def evaluate(mnist):
    x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")

    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    y = mnist_inference.inference(x,None)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 将布尔型转为实数型，再计算平均值
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    #通过变量重命名来加载模型，这样在前向传播的过程中就不用调用滑动平均函数来获取平均值了
    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variable_averages_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variable_averages_restore)
    merged = tf.summary.merge_all()
    while True:
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("Log/log", tf.get_default_graph())
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                #通过模型文件名得到迭代轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                accuracy_score = sess.run(accuracy,val_feed)
                tf.summary.scalar('accuracy_score', accuracy_score)

                print("after %d training steps,accuracy on validate is %g", global_step, accuracy_score)

            else:
                print('no check point file found')
                return
        time.sleep(EVAL_SECOND)


def main(argv=None):
    mnist = downloadMNIST.read_data_sets("MNIST_data/", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()




