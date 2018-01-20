import tensorflow as tf
import numpy as np
import random
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

"""
'I'm super man'
tokenize:
['I', ''m', 'super','man' ] 
"""
from nltk.stem import WordNetLemmatizer

"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""

pos_file = 'data/pos.txt'
neg_file = 'data/neg.txt'

# INPUT_NODE = len(lex) #网络输入节点数 就是整个词库的大小，因为我们的提取样本特征是这么长[0,1,0,0,1....]
INPUT_NODE = 1065
OUTPUT_NODE = 2

LAYER1_NODE = 500
LAYER2_NODE = 500

batch_size = 100
epochs = 20

#基础学习率，学习率的衰减率
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

def process_file(file):
    with open(file, 'r') as f:
        lex = []
        lines = f.readlines()
        for line in lines:
            words = word_tokenize(line.lower())
            # [1,2]+[3,4]为[1,2,3,4]。同extend()
            lex += words
        return lex


def create_lexicon(pos_file, neg_file):
    lex = []
    lex += process_file(pos_file)
    lex += process_file(neg_file)
    # print(len(lex))
    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]  # 词形还原 (cats->cat)

    word_count = Counter(lex)
    # print(word_count)
    # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:  # 这写死了，好像能用百分比
            lex.append(word)  # 齐普夫定律-使用Python验证文本的Zipf分布 http://blog.topspeedsnail.com/archives/9546
    return lex


# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
def normalize_dataset(lex):
    dataset = []

    # lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review, clf):
        # print(type(clf))
        words = word_tokenize(line.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]   # 词形还原 (cats->cat)

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        # print(type(np.array(clf)))
        return [features, np.array(clf)]

    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1, 0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
            np.array(one_sample)
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0, 1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
            np.array(one_sample)
            dataset.append(one_sample)

    # print(len(dataset))
    return dataset

def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection("losses",regularizer(weights))

    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases", shape=[LAYER1_NODE])
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE,LAYER2_NODE],regularizer)
        biases =tf.get_variable("biases",shape=[LAYER2_NODE])
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    with tf.variable_scope("layer3"):
        weights = get_weight_variable([LAYER2_NODE,OUTPUT_NODE],regularizer)
        biases =tf.get_variable("biases",shape=[OUTPUT_NODE])
        layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)

    return layer3

def train(dataset):
    # 取样本中的10%做为测试数据
    test_size = int(len(dataset) * 0.1)
    dataset = np.array(dataset)
    # print(type(dataset))
    # print(dataset.shape)
    # print(dataset)

    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]
    # print(train_dataset)

    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    y = inference(x,None)

    global_step = tf.Variable(0, trainable=False)

    #准确度
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean

    # 设置指数衰减学习率
    # learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 100, LEARNING_RATE_DECAY)

    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy_mean,global_step=global_step)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        random.shuffle(train_dataset)
        train_x = train_dataset[:,0]
        train_y = train_dataset[:,1]

        for epoch in range(epochs):
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                # for i in range(len(batch_x)):
                #     print(batch_x[i])
                #     print(batch_y[i])
                _,value_loss,step = sess.run([optimizer, loss, global_step],feed_dict={x:list(batch_x),y_:list(batch_y)})

                if i % 50 == 0:
                    print("after %d training steps,loss on training is %g" % (step,value_loss))

                i += batch_size

        text_x = test_dataset[:, 0]
        text_y = test_dataset[:, 1]
        accuracy = sess.run(accuracy,feed_dict={x:list(text_x),y_:list(text_y)})

        print('准确率: ', accuracy)



def main(argv=None):

    dataset = None
    try:
        with open('data/save.pickle', 'rb') as f:
            dataset = pickle.load(f)
    except IOError as error:
        print('File Error: ' + str(error))
        # lex里保存了文本中出现过的单词。作为单词表  len(lex) = 1065
        lex = create_lexicon(pos_file, neg_file)
        dataset = normalize_dataset(lex)
        random.shuffle(dataset)
        # print(dataset)
        # 把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
        with open('data/save.pickle', 'wb') as f:
            pickle.dump(dataset, f)
    finally:
        train(dataset)


if __name__ == '__main__':
    tf.app.run()
