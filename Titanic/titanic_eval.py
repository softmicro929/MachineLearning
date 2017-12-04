import tensorflow as tf
import time
import pandas as pd
import numpy as np

from Titanic import titanic_data
from Titanic import titanic_inference
from Titanic import titanic_train

EVAL_SECOND = 10


def evaluate(test_data):
    x = tf.placeholder(tf.float32, [None, titanic_inference.INPUT_NODE], name="x-input")

    # 用正则取出我们要的属性值
    test_df = test_data.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    test_np = test_df.as_matrix()

    # X即特征属性值
    X = test_np[:, 0:]

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
                {'PassengerId': test_data['PassengerId'].as_matrix(), 'Survived': predictions})
            result.to_csv("./data/prediction.csv", index=False)

            print("the prediction is created")

        else:
            print('no check point file found')
            return


def main(argv=None):
    data_url = "./data"
    train_data, test_data = titanic_data.read_data_sets_from_csv(data_url)
    evaluate(test_data)


if __name__ == '__main__':
    tf.app.run()
