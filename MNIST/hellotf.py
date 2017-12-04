import tensorflow as tf

from MNIST import downloadMNIST

mnist = downloadMNIST.read_data_sets("MNIST_data/", one_hot=True)

print("the train set size:",mnist.train.num_examples)
print("the validate set size:",mnist.validation.num_examples)
print("the test set size:",mnist.test.num_examples)

batch_size = 100
xs,ys = mnist.train.next_batch(batch_size)

print("xs shape:",xs.shape)
print("ys shape:",ys.shape)

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))