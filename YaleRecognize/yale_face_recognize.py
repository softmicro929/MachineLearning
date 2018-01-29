import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    img_mat = np.reshape(img_arr, (64, 64))
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


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# def show_cnn_image(conv, x):
#     result = conv.eval(feed_dict=)
#     for _ in xrange(32):
#         show_img = result[:, :, :, _]
#         show_img.shape = [28, 28]
#         plt.subplot(4, 8, _ + 1)
#         plt.imshow(show_img, cmap='gray')
#         plt.axis('off')
#     plt.show()

# 前向卷积过程
def inference(images):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 1, 32],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # LRN全称是local response normalization，局部响应归一化
    # LRN是normalization的一种，normalizaiton的目的是抑制，抑制神经元的输出。
    # 而LRN的设计借鉴了神经生物学中的一个概念，叫做“侧抑制”。
    # 从公式和历史由来可以看出，LRN是对channel做计算的。
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 32, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        pool2_shape = pool2.get_shape().as_list()
        nodes = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
        # reshape = tf.reshape(pool2, [batch_size, nodes])

        # print(nodes)
        # print(pool2_shape[0])

        reshape = tf.reshape(pool2, [-1, nodes])

        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, num_class],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [num_class],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_linear, conv1, conv2


# 训练
def train(train_data, train_labels, test_data, test_labels):
    x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 15])

    y, conv1, conv2 = inference(x)

    global_step = tf.Variable(0, trainable=False)

    # 准确度
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=y)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean

    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy_mean, global_step=global_step)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for epoch in range(epochs):
            i = 0
            while i < len(train_data):
                start = i
                end = i + batch_size

                batch_x = train_data[start:end]
                # batch_x = np.reshape(batch_x, (None,64,64,1))
                batch_y = train_labels[start:end]

                # print(np.shape(train_data[start:end]))
                # print(type(train_data[start:end]))
                # batch_x = np.array(train_data[start:end])
                # print(np.shape(batch_x))
                # print(type(batch_x))

                _, value_loss, step = sess.run([optimizer, loss, global_step], feed_dict={x: batch_x, y_: batch_y})

                if i % 50 == 0:
                    print("--------------> %d training steps,loss : %g" % (step, value_loss))

                i += batch_size

            accu = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
            print("======================================================>%d epoch,accuracy : %g" % (epoch, accu))

            if (epoch):
                # 这里我需要一个tensor：conv1
                X_img = train_data[0:1]
                Y_img = train_labels[0:1]

                result = conv1.eval(feed_dict={x: X_img, y_: Y_img})
                # result2 = conv2.eval(feed_dict={x: X_img, y_: Y_img})

                # print(result.shape)
                # print(type(result))
                for _ in range(32):
                    show_img = result[:, :, :, _]
                    show_img.shape = [64, 64]
                    plt.subplot(4, 8, _ + 1)
                    plt.imshow(show_img, cmap='gray')
                    plt.axis('off')
                plt.show()
                # for _ in range(64):
                #     show_img2 = result2[:, :, :, _]
                #     show_img2.shape = [32, 32]
                #     plt.subplot(8, 8, _ + 1)
                #     plt.imshow(show_img2, cmap='gray')
                #     plt.axis('off')
                # plt.show()


def main(argv=None):
    try:
        train_data, train_labels = read_file('data/Yale_64x64.mat')
        if (train_data is None or train_labels is None):
            return
        print(type(train_data))
        print(type(train_labels))

        show_img(train_data[0])

        train_data, train_labels, test_data, test_labels = process_data(train_data, train_labels)

        print(np.shape(train_data))

        train(train_data, train_labels, test_data, test_labels)

    except IOError as error:
        print('IO ERROR ' + str(error))


if __name__ == '__main__':
    tf.app.run()
