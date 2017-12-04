import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name="v2")

result = v1 + v2

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    saver.export_meta_graph("./model/model.ckpt.meta.json",as_text=True)
    #导出计算图中 graph_def 部分
    graph_def = tf.get_default_graph().as_graph_def()

    output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,["add"])

    with tf.gfile.GFile("./model/combine_model.pb","wb") as f:
        f.write(output_graph_def.SerializeToString())