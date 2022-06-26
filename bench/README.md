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

Install dependencies (from bench/)
```
pip install click gpflow memory_profiler && \
pip install git+https://github.com/hughsalimbeni/bayesian_benchmarks@master && \
cp -r ./uci /usr/local/lib/python3.6/dist-packages/bayesian_benchmarks/data/
```

XLA_FLAGS="--xla_tensor_size_threshold=7GB --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/ --xla_enable_hlo_passes_only=tensor-splitter,broadcast-simplifier,dot-order-optimizer,dce,flatten-call-graph" python ./bench.py --warmup 1 --repeat 1 --logdir "./logs/kernel-vector-product/test" -f fp64 sgpr -d houseelectric
