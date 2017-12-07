import tensorflow as tf
import time

from Titanic import titanic_data
from Titanic import titanic_inference
from Titanic import titanic_train

EVAL_SECOND = 10


def evaluate(validate_data):
    x = tf.placeholder(tf.float32, [None, titanic_inference.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, titanic_inference.OUTPUT_NODE], name="y-input")

    # X即特征属性值
    X = validate_data.get_x()
    Y = validate_data.get_y()

    val_feed = {x: X, y_: Y}

    y = titanic_inference.inference(x, None)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 将布尔型转为实数型，再计算平均值
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 通过变量重命名来加载模型，这样在前向传播的过程中就不用调用滑动平均函数来获取平均值了
    variable_averages = tf.train.ExponentialMovingAverage(titanic_train.MOVING_AVERAGE_DECAY)
    variable_averages_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variable_averages_restore)

    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(titanic_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过模型文件名得到迭代轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                accuracy_score = sess.run(accuracy, val_feed)
                tf.summary.scalar('accuracy_score', accuracy_score)

                print("after %s training steps,accuracy on validate is %g"%(global_step, accuracy_score))

            else:
                print('no check point file found')
                return
        time.sleep(EVAL_SECOND)

def main(argv=None):
    data_url = "./data"
    validate_data = titanic_data.read_validation_data_sets(data_url)
    evaluate(validate_data)


if __name__ == '__main__':
    tf.app.run()
