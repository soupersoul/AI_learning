import tensorflow as tf
import input_data as indata

mnist = indata.read_data_sets("DataSets/", one_hot = True)

def placeholder_inputs(batch_size):
    images_holder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_holder = tf.placeholder(tf.int32, shape = [batch_size])
    return (images_holder, labels_holder)

#x = tf.placeholder(tf.float32, [None, 784])
#y = tf.placeholder(tf.float32, [None, 10])

def weight(shape)
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def biases(shape)
    return tf.Variable(tf.constant(1.0, shape))
