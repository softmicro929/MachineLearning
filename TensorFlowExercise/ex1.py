import tensorflow as tf

a = tf.constant([[1,2]],name="a",dtype=tf.float32)
print(a)
b = tf.constant([[2.0],[3.0]],name="b")
print(b)
product = tf.matmul(a,b)

session = tf.Session()

result = session.run(product)

print(product)

print(result)

session.close()
