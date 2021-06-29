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