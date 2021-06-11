from examples import cases
import tensorflow as tf
import tensorflow.config.experimental as tf_exp
import sys
from time import time

case_name = sys.argv[1]
size = int(sys.argv[2])
for case in cases:
  if case.__name__ == case_name:
    break

args = [tf.random.normal(tuple(size * a for a in arg)) for arg in case.arguments]

start_unix = time()
case.run(*args)

# Collect time and memory stats ...
runtime = time() - start_unix
memory = tf_exp.get_memory_info('GPU:0')['peak']

print(f"{runtime}\t{memory}")
exit(0)
