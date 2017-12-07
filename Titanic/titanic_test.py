import tensorflow as tf
import pandas as pd

from Titanic import titanic_data
from Titanic import titanic_inference
from Titanic import titanic_train

EVAL_SECOND = 10


def evaluate(test_data):
    x = tf.placeholder(tf.float32, [None, titanic_inference.INPUT_NODE], name="x-input")

    # X即特征属性值
    X = test_data.get_x()[:, 0:]

    val_feed = {x: X}

    y = titanic_inference.inference(x, None)

    predictions = tf.argmax(y, 1)

    # 通过变量重命名来加载模型，这样在前向传播的过程中就不用调用滑动平均函数来获取平均值了
    variable_averages = tf.train.ExponentialMovingAverage(titanic_train.MOVING_AVERAGE_DECAY)
    variable_averages_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variable_averages_restore)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(titanic_train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 通过模型文件名得到迭代轮数
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            predictions = sess.run(predictions, feed_dict=val_feed)

            print(predictions.shape)
            print(predictions)
            result = pd.DataFrame(
                {'PassengerId': pd.read_csv("./data/test.csv")['PassengerId'].as_matrix(), 'Survived': predictions})
            result.to_csv("./data/prediction.csv", index=False)

            print("the prediction.csv is created")

        else:
            print('no check point file found')
            return


def main(argv=None):
    data_url = "./data"
    test_data = titanic_data.read_test_data_sets(data_url)
    evaluate(test_data)


if __name__ == '__main__':
    tf.app.run()
