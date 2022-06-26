Gambit Project
===========================

_Authors: Artem Artemev, Tilman Roeder, and Mark van der Wilk_

XLA is a compiler for linear algebra. Frameworks - PyTorch, TensorFlow and JAX support it in some way. XLA is an obvious choice for optimization tweaks in user-defined expressions (computational graphs) implicitly without user interventions.

### Evaluation order

Let consider a simple math expression that involves single matrix-matrix multiplication and matrix-vector multiplication. For a given N-by-M matrix $A$, M-by-D matrix $B$, and D-by-1 vector $v$, we have the following expression:

$$
c =  (A \times B) \times v
$$

Resulting vector $c$ has dimensionality N-by-1. The result in the parenthesis equals to N-by-D matrix, and the cost of that operation is $\mathcal{O}(NMD)$. Following vector multiplication costs another $\mathcal{O}(ND)$ and gives N-by-1 $c$ vector.

This order of execution is neither memory nor CPU/GPU clock efficient. A better choice would be to traverse the computational graph a bit differently:

$$
c =  A \times (B \times v)
$$

$B \times v$ matrix-vector multiplication costs $\mathcal{O}(MD)$ with a temporary $\mathcal{O}(M)$ memory footprint. Next in turn is again the vector-matrix multiplication with $\mathcal{O}(NM)$ cost.

The conclusion is:

- Perform vector multiplication first - it is always cheaper
- By changing the order of matrix operations, we can speed up algorithms and save memory in intermediate steps of expression.

### Operation transformations

Euclidean distance in $\mathbb{R}^K$ space between $A$ and $B$ vector collections with lengths $N$ and $M$ respectively, such that $A$ is a N-by-K matrix and $B$ is a M-by-K matrix:
$$
D_{i,j} = ||A_{i,:} - B_{j,:}||^ 2
$$

such that, $D$ is a N-by-M distance matrix and $D_{i,j}$ element is a euclidean distance between $i$-th row in $A$ and $j$-th row in $B$. Please, note that in the equation above, the $A - B$ executes according to the NumPy broadcasting rules:

```python
import numpy as np

N: int = 3
M: int = 5
K: int = 2

A: np.ndarray = np.random.randn(N, K)
B: np.ndarray = np.random.randn(M, K)

C = (A[:, np.newaxis, :] - B[np.newaxis, ...]) **  2
D = np.sum(C, axis=-1)
```

Intermediate $C$ tensor has a shape [N, M, K] that stresses a memory requirement for devices. Particularly, the naive euclidean computation becomes very expensive to run on GPU.

An alternative to naive computation would be an observation that the distance has a quadratic form:

$$
D_{i,j} = ||A_{i,:}||^2 + ||B_{j,:}||^2 - 2 A_{i,:}^T B_{j,:}
$$

The expression boils down to a matrix-matrix product between transposed matrix $A$ and matrix $B$:

```python
D = np.sum(A ** 2, axis=-1)[np.newaxis, :] + \
    np.sum(B ** 2, axis=-1)[:, np.newaxis] - \
    2 * A.T @ B
```

### Map-Reduce and lazy evaluations

JAX, TensorFlow and PyTorch offer evaluations on GPU and CPU devices with fully materialized tensors. If user's program cannot allocate memory for a tensor, usually it crashes with out-of-memory error (OOM). User could prevent the OOM behavior by splitting arrays into slices (blocks or tiles), treating these slices independently, and evaluating operations in a lazy and distributed manner, engaging all available devices.
If an operation cannot be applied to all slices at once, a user can decide to cache slices and run the operation sequentially on a subset. Of course, the latter approach might run slower. However, the benefit of that approach is that the code would be feasible to run even under hard resource constraints.

Matrix multiplication is a perfect example for a map-reduce scheme. For a given matrix $A$, N-by-D size, and a matrix $B$, D-by-M size, matrix multiplication as mentioned earlier costs $\mathcal{O}(NDM)$ and because each element $C_{i,j}$ of the output matrix $C = A \times B$ is independent of other elements, this operation is highly parallelizable. GPU-accelerated libraries CUDA and MAGMA have super-efficient implementation for this op. The pitfall is in a GPU memory limitation. For matrices with large N, D or M temporary computations will not fit into the GPU memory.


## Notes on building TF/XLA

Basically follow the steps at https://www.tensorflow.org/install/source?hl=en#docker_linux_builds (use Docker on linux, otherwise the build will take forever, since docker on MacOS is running in a VM; note that the build will take around 2-5 hours)

1. Clone the gambit repository:
    ```bash
    git clone git@github.com:awav/gambit.git
    cd gambit
    git submodule init && git submodule update
    ```
2. Get the docker image:
    ```bash
    docker pull tensorflow/tensorflow:devel
    ```
3. Run the docker container. Inside `gambit` run:
    ```bash
    docker run -it -w /mnt -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:devel bash
    ```
4. Make sure to set up the bazel cache directory!
    - Set up a `.cache` folder inside of the cloned gambit: `mkdir .cache`
    - After starting the docker container, symlink it: `ln -s /mnt/.cache /root/.cache`
    - Make sure to set the bazel cache directory to within the mounted files, so they are not lost when you restart your container.
    - If you forgot this, this can be fixed after the first build by running: `cp /root/.cache /mnt/.cache`
5. For the first build configure the project: run `./configure` inside the `tensorflow` directory.
6. Run the build inside the `tensorflow` directory. Expect the first run to take between 2-5 hours:
    ```bash
    # building the pip package
    bazel build //tensorflow/tools/pip_package:build_pip_package
    # running (only our) XLA tests
    bazel test //tensorflow/compiler/xla/service:dot_order_optimizer_test
    bazel test //tensorflow/compiler/xla/service:tensor_splitter_test
    # build and install pip package
    bazel build //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /mnt
    pip install ../tensorflow-2.5.0-cp36-cp36m-linux_x86_64.whl -U
    # build and install nightly pip package
    bazel build //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /mnt
    pip install ../tf_nightly-2.5.0-cp36-cp36m-linux_x86_64.whl -U
    # all one cmd
    bazel build //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /mnt && \
    pip install ../tensorflow-2.5.0-cp36-cp36m-linux_x86_64.whl -U
    ```
7. Extract images from XLA and other options:
    ```bash
    # All passes
    TF_DUMP_GRAPH_PREFIX="./xla-dump/" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/ --xla_tensor_size_threshold=1GB" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices --tf_xla_clustering_debug" python xla_playground.py
    # Only our pass
    TF_DUMP_GRAPH_PREFIX="./xla-dump/" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/ --xla_enable_hlo_passes_only=tensor-splitter,broadcast-simplifier,dot-order-optimizer,dce --xla_tensor_size_threshold=1GB" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices --tf_xla_clustering_debug" python xla_playground.py
    # Disable our hlo pass
    XLA_FLAGS="--xla_disable_hlo_passes=tensor-splitter" python ...
    # Option for setting the split sizes threshold
    TF_DUMP_GRAPH_PREFIX="./xla-dump/" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/ --xla_tensor_size_threshold=2000000" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices --tf_xla_clustering_debug" python xla_playground.py
    ```
8. Run benchmarks:
    ```bash
    # Install dependencies (for CPU profiler)
    pip install memory_profiler
    # Run with our pass
    python bench/main.py "bench_with_split.csv"
    # Run without our pass
    XLA_FLAGS="--xla_disable_hlo_passes=tensor-splitter" python bench/main.py "bench_no_split.csv"
    ```
9. Notes:
    - If you need the physical splitting in the graph (separate nodes as opposed to while loops) use this commit:
      <https://github.com/awav/tensorflow/commit/304ad922091bc672b0c0d7017260fb24d4267d23>
    - See why this wont be split ... :
    ```
    XLA_FLAGS="--xla_tensor_size_threshold=1GB --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/" python ./bench.py --warmup 1 --repeat 1 --logdir "./logs/kernel-vector-product/test" -f fp64 kernel-vector-product -k se -a "(100000, 10)" -b "(100000, 10)" -v "(100000, 1)"
    ```


## TensorFlow local compiling

1. 
    ```bash
    git clone git@github.com:awav/gambit.git
    cd gambit
    git submodule init && git submodule update
    ```

2. Install bazelisk https://github.com/bazelbuild/bazelisk/releases
    Install it as the bazel binary in your `PATH` (e.g. copy it to `/usr/local/bin/bazel`). Never worry about upgrading Bazel to the latest version again.

3. Pip installations (surprize, surprize!)
    ```
    pip install -y numpy keras_preprocessing
    ```

3. Local installation (CUDA)
    ```
    DEV=cuda
    TF_PIP_PATH=~/Storage/tf-pip
    rm -rf $TF_PIP_PATH &&
    bazel build //tensorflow/tools/pip_package:build_pip_package --config=$DEV &&
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package $TF_PIP_PATH &&
    pip uninstall -y tensorflow tensorflow-estimator &&
    pip install -U $TF_PIP_PATH/tensorflow-*.whl
    ```

## JAX local compiling

1. GPU:
    ```
    CUDA_VERSION=11.2
    JAX_DIST=~/code/jax/dist
    rm -rf $JAX_DIST/jaxlib-*.whl &&
    python build/build.py --enable_cuda --cuda_version=$CUDA_VERSION &&
    pip install --force-reinstall $JAX_DIST/jaxlib-*.whl &&
    pip install -e .
    ```

## Building with JAX
1. Download JAX repo: `git clone https://github.com/google/jax.git`
2. Check out a compatible version: `git checkout 8c3371c`
3. Set the modified version of tensorflow in the file `WORKSPACE` in JAX repo
```
# (comment out the http archive)

# For development, one can use a local TF repository instead.
local_repository(
   name = "org_tensorflow",
   path = "/mnt/tensorflow",
)
```
4. Run the build: `python build/build.py`
5. Follow the instructions on screen to install the built wheel for jaxlib
6. Install jax: `pip install -e .` 
