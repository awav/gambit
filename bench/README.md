Gambit benchmarks 
===

The main benchmark script is `bench.py`:

```bash
→ python ./bench.py --help
Usage: bench.py [OPTIONS] COMMAND [ARGS]...

Options:
  -f, --float-type FLOAT_TYPE
  -s, --seed INTEGER
  -r, --repeat INTEGER         Number of experiment repeatitions
  -w, --warmup INTEGER         Number of warm-up iterations
  -l, --logdir PATH
  --xla / --no-xla             Compile function with or without XLA
  -b, --backend [tf|jax]       TensorFlow or JAX framework
  --help                       Show this message and exit.

Commands:
  dist
  kvp
  kvp-grads
  matrix-chain
  outerprod
  sgpr
  tril-solve
```

The `bench.py` collects simple statistics and traces of the requested benchmark, saves them in `stats.npy` file and locates it in specified by user logging directory.

We use [`xpert`](https://github.com/awav/xpert) to run cross product configuration benchmarks. Check `splitting-benchmark.toml` configuration file.

Install dependencies (from bench/)
```
pip install click gpflow memory_profiler && \
pip install git+https://github.com/hughsalimbeni/bayesian_benchmarks@master && \
cp -r ./uci /usr/local/lib/python3.6/dist-packages/bayesian_benchmarks/data/
```

XLA_FLAGS="--xla_try_split_tensor_size=7GB --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/ --xla_enable_hlo_passes_only=algebraic-rewriter,broadcast-simplifier,dot-order-optimizer,rce-optimizer,tensor-splitter,mco,dce,flatten-call-graph" python ./bench.py --warmup 1 --repeat 1 --logdir "./logs/kernel-vector-product/test" -f fp64 sgpr -d houseelectric
