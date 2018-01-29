import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

learning_rate = 0.0001
batch_size = 5
num_class = 15
epochs = 10


# 加载数据
def read_file(filename):
    try:
        with open(filename, 'rb') as f:
            dict = sio.loadmat(f)
        return dict['fea'], dict['gnd']
    except IOError as error:
        print('file open error', str(error))


# 显示一张图片
def show_img(img_arr):
    img_mat = np.reshape(img_arr, (128, 128))
    # img = Image.fromarray(img_mat)
    # img.show()
    plt.imshow(img_mat, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()


# 知道了，np.zeros生成返回的one_hot矩阵列表，numlables*num_classes(165*15)
# 然后将每一行对应分类置为1，怎么找到每行的位置呢，使用index_offset+类别号（0-14）
def label_to_one_hot(labels_dense, num_classes=15):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# 将图片转灰度图
def to4d(img):
    return img.reshape(img.shape[0], 64, 64, 1).astype(np.float32) / 255


# 将数据shuffle随机，并在训练集上切分出测试集
# 将类别转为one_hot
# 将图像转为灰度图
def process_data(train_data, train_labels):
    # 将label转为0-14
    train_labels = train_labels - 1

    # shuffle data
    np.random.seed(100)
    train_data = np.random.permutation(train_data)

    np.random.seed(100)
    train_labels = np.random.permutation(train_labels)

    test_data = train_data[0:50, :]
    test_labels = train_labels[0:50]

    np.random.seed(200)
    test_data = np.random.permutation(test_data)
    np.random.seed(200)
    test_labels = np.random.permutation(test_labels)

    train_data = to4d(train_data)
    train_labels = label_to_one_hot(train_labels, 15)
    test_data = to4d(test_data)
    test_labels = label_to_one_hot(test_labels, 15)

    return train_data, train_labels, test_data, test_labels


def read_tfrecord(filename):
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    img = tf.reshape(img, [64, 64, 1])  # reshape为128*128的3通道图片
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    return img, label


def write_tfrecord(features, labels, record_name):
    writer = tf.python_io.TFRecordWriter(record_name)  # 要生成的文件

    num_examples = features.shape[0]
    print(num_examples)

    for index in range(num_examples):
        # print(features[index])
        print('feature ', index, ' shape ' ,np.shape(features[index]))
        img_raw = features[index].tostring()  # 将图片转化为二进制格式
        print('write img in ', index)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[np.argmax(labels[index])])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))  # example对象对label和image数据进行封装

        writer.write(example.SerializeToString())  # 序列化为字符串
        print('   writer', index, 'DOWN!')

    writer.close()
    return record_name


def create_record():
    try:
        train_data, train_labels = read_file('data/Yale_64x64.mat')
        print(type(train_data))
        print(type(train_labels))

        if (len(train_data) > 0 and len(train_labels) > 0):
            show_img(train_data[0])

            train_data, train_labels, test_data, test_labels = process_data(train_data, train_labels)

            # (165, 64, 64, 1)
            # (64, 64, 1)
            # (15,)
            print(np.shape(train_data))
            print(np.shape(train_data[0]))
            print(np.shape(train_labels[0]))
            print(train_labels[0])
            print(np.argmax(train_labels[0]))

            record_train = write_tfrecord(train_data, train_labels, 'data/yale_train.tfrecords')
            # record_test = write_tfrecord(test_data, test_labels, 'data/yale_test.tfrecords')

    except IOError as error:
        print('IO ERROR ' + str(error))


# the program start
if __name__ == '__main__':
    # create_record()
    filename_queue = tf.train.string_input_producer(['data/yale_train.tfrecords'])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    # img = tf.image.resize_images(img, (64, 64, 1))
    # img.set_shape(tf.pack([64, 64, 1]))
    # print(img)
    print('----------------')
    img = tf.reshape(img, [128, 128, 1])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量

    # print(img)
    print(label)

    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=5,
                                capacity=100, min_after_dequeue=50, num_threads=2)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        img_batch, label_batch = sess.run([img_batch, label_batch])
        print(np.shape(img_batch))

        for i in range(5):
            show_img(img_batch[i])
        print(np.shape(label_batch))
        coord.request_stop()
        coord.join(threads)
