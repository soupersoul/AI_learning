import tensorflow as tf

state = tf.Variable(0, name="Counter")
delta = tf.constant(1)
result = tf.add(state, delta)
update = tf.assign(state, result)

#init_op = tf.initialize_all_variables()  deprecated
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(state)
    for _ in range(3):
        sess.run(update)
        print sess.run(state)
