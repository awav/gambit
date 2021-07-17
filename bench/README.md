Gambit benchmarks 
===

The main benchmark script is `bench.py`:

```bash
→ python ./bench.py --help
Usage: bench.py [OPTIONS] COMMAND [ARGS]...

Options:
  -f, --float-type FLOAT_TYPE
  -m, --mem-limit MEMORY_LIMIT
  -s, --seed INTEGER
  -r, --repeat INTEGER
  -w, --warmup INTEGER
  -l, --logdir PATH
  --xla / --no-xla
  --help                        Show this message and exit.

Commands:
  kernel-vector-product
  outerprod
  outerprod-with-itself
```

The `bench.py` collects simple statistics and traces of the requested benchmark, saves them in `stats.npy` file and locates it in specified by user logging directory.

We use [`xpert`](https://github.com/awav/xpert) to run cross product configuration benchmarks. Check `splitting-benchmark.toml` configuration file.


TF_DUMP_GRAPH_PREFIX="./xla-dump/" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/ --xla_try_split_tensor_size=4000000" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices --tf_xla_clustering_debug" python bench.py -f fp64 -s 0 -r 10 -w 2 -l logs --xla kernel-vector-product -k se -a 1000,100 -v 1000,10
