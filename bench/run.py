from examples import cases
import tensorflow as tf
import tensorflow.config.experimental as tf_exp
import sys
from time import time

HAS_GPU = len(tf.config.list_physical_devices('GPU'))

case_name = sys.argv[1]
size = int(sys.argv[2])
for case in cases:
  if case.__name__ == case_name:
    break

args = tuple(tf.random.normal(tuple(size * a for a in arg)) for arg in case.arguments)

# Collect time and memory stats ...
if HAS_GPU:
  start_unix = time()
  case.run(*args)
  runtime = time() - start_unix
  memory = tf_exp.get_memory_info('GPU:0')['peak']
else:
  from memory_profiler import memory_usage
  start_unix = time()
  mem = memory_usage((case.run, args, {}))
  runtime = time() - start_unix
  memory = max(mem)

print(f"{runtime}\t{memory}")
exit(0)
