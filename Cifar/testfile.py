import tensorflow as tf
import os,sys

filenames = ['tmp/cifar10_data/cifar-10-batches-bin/data_batch_1.bin', 'tmp/cifar10_data/cifar-10-batches-bin/data_batch_2.bin', 'tmp/cifar10_data/cifar-10-batches-bin/data_batch_3.bin', 'tmp/cifar10_data/cifar-10-batches-bin/data_batch_4.bin', 'tmp/cifar10_data/cifar-10-batches-bin/data_batch_5.bin']
print(sys.path[0])
for f in filenames:

    print(sys.path[0])
    print(f)

    if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)
    else:
        print(f + " is exits")