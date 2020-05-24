import tensorflow_datasets as tfds
import tensorflow as tf
import input_data
#import tensorflow.examples.tutorials.mnist.input_data as input_data
#mnist = tfds.load('mnist')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_=tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
