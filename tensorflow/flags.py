
# usage: python flags.py --var_str=abc --var_int=123 --var_bool=False
#        python flags.py --var_str abc --var_int 123 --var_bool False
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("var_str", "string_val", "this is my string")
flags.DEFINE_int("var_int", 1, "this is my int")
flags.DEFINE_bool("var_bool", True, "this is my bool" )

def main():
  print(flags.FLAGS.var_str)
  print(flags.FLAGS.var_int)
  print(flags.FLAGS.var_bool)

if __name__ == '__main__':
  tf.app.run()
