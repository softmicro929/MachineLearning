import tensorflow as tf

INPUT_NODE = 17
OUTPUT_NODE = 2

LAYER1_NODE = 500
LAYER2_NODE = 500

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

    # with tf.variable_scope("layer3"):
    #     weights = get_weight_variable([LAYER2_NODE,OUTPUT_NODE],regularizer)
    #     biases =tf.get_variable("biases",shape=[OUTPUT_NODE])
    #     layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)


    return layer2