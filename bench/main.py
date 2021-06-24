from examples import cases
import subprocess

# TODO: Get from cmd ...
SIZES = [1, 2, 8, 16, 32]# , 64, 128]

def run_case(case):
  self_dir = "/".join(__loader__.path.split("/")[:-1])
  stats = []
  for size in SIZES:
    # run in process such that the peak memory use is isolated
    print(f"| running size = {size}")
    process = subprocess.Popen(["python3", f"{self_dir}/run.py", case.__name__, f"{size}"],
                               shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.stdout.read().decode("utf-8")
    stderr = process.stderr.read().decode("utf-8")
    try:
      runtime, memory = output.split("\t")
      stats.append({
        "size": size,
        "runtime": float(runtime),
        "memory": float(memory),
      })
    except:
      print("\nAn error occured:")
      print("=================\n")
      print(stderr)
      exit(1)
  return stats

print("Running benchmark cases ...")
output_rows = []
for case in cases:
  print(f"Benchmark case {case.__name__}")
  for stats in run_case(case):
    output_rows.append(
      # CSV row: case, size, runtime, peak memory
      [f"case:{case.__name__}", stats['size'], stats['runtime'], stats['memory']]
    )

# Output CSV
csv_txt = "\n".join(["\t".join([str(cell) for cell in row]) for row in output_rows])
open("bench.csv", "w").write(csv_txt)
