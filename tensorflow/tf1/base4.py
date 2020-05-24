import tensorflow as tf

sess = tf.InteractiveSession()

state = tf.Variable(1, name="Counter")
delta = tf.constant(2)
new_value = tf.add(state, delta)
update = tf.assign(state, new_value)

state.initializer.run()
for _ in range(3):
    print update.eval()
