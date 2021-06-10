from examples import cases
import tensorflow as tf

# TODO: Get from cmd ...
SIZES = [1, 2, 8, 16, 32]# , 64, 128]

def run_case(case):
  for size in SIZES:
    print(f"| running size {size}")
    args = [tf.random.normal(tuple(size * a for a in arg)) for arg in case.arguments]
    # TODO: Reset tfl stats
    case.run(*args)
    # TODO: Collect tfl stats

print("Running benchmark cases ...")
for case in cases:
  print(f"Benchmark case {case.__name__}")
  run_case(case)

# TODO: Output tfl stats
