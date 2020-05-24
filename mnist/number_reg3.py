import tensorflow_datasets as tfds
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
def norm(x):
    return tf.nn.local_response_normalization(x)
x = tf.placeholder("float", shape=[None, 784])
y_=tf.placeholder("float", shape=[None, 10])
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(norm(conv2d(x_image, w_conv1) + b_conv1))
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(norm(conv2d(h_pool1, w_conv2) + b_conv2))
h_pool2 = max_pool_2x2(h_conv2)

w_f1 = weight_variable([7*7*64, 1024])
b_f1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_f1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_f1) + b_f1)

keep_prob = tf.placeholder(tf.float32)
h_f1_prob = tf.nn.dropout(h_f1, keep_prob)

W = weight_variable([1024, 10])
b = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_f1_prob, W) + b)




# y_*tf.log(y): tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) # or tf.reduce_mean(....)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob:0.5})

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))