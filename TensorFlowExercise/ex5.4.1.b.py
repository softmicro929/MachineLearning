import tensorflow as tf

from tensorflow.python.platform import gfile

#from tensorflow.python.framework import graph_util
#这里声明的变量名要和saver存储的一样，不一样会报找不到，或者可以做名字映射，在saver里
# v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
# v2 = tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
#
# result = v1 + v2
#
# saver = tf.train.import_meta_graph("./model/model.ckpt.meta")

with tf.Session() as sess:
    # saver.restore(sess,"./model/model.ckpt")
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
    model_filename = './model/combine_model.pb'
    #读取保存的模型文件，解析成GraphDef
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #将graph_def保存的图加载到当前图中，return_elements给出了返回张量的名字
    #在保存的时候是给出的计算节点的名字"add"，在加载时给出的是张量的名称"add:0"
    #没太明白？？？这个名字怎么定义，加0？？？
    result = tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(result))