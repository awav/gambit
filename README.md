Draft of the Gambit project
====

_Authors: Artem Artemev (art.art.v@gmail.com), Tilman Roeder (tilmroe@gmail.com), and continues..._

> _Up-to-date description is here (with latex rendering):
> https://colab.research.google.com/drive/1TyrctD_Cv2p6f-wtoqgv5Oq60pfcKSmE?usp=sharing_

## Motivation

Scientific programming has multiple challenges that come from the plexus of various disciplines. The focal point of this document is Machine Learning (ML) - an intersection of mathematics and computer science. ML success solely relies on the performance and efficiency of a code that involves linear algebra and complicated math. Writing adequate code that is fast and scales well using standard computer programming languages, like C++ or Python (popular choices), is a difficult task. When efficiency is mandatory for mathematical algorithms, the code complexity ramps up. And in turn, demands from a developer to have sharply honed skills in distributed and parallel computing.

Today, frameworks like TensorFlow, JAX and PyTorch provide math abstractions with automatic differentiation capabilities, and on top of that, they give their users GPU acceleration for free. More sophisticated algorithms claim more hardware resources, and when this requirement is met, those frameworks don't facilitate scaling algorithms with multiple GPUs. They don't evaluate operations lazily in a memory-efficient manner. However, lots of linear algebra operations and math expressions are distributable and can be evaluated in memory saving mode.

## Mission

Implement MLIR (XLA) compiler extensions for optimizing linear algebra and math expressions. Anyone who uses this extension: mathematician, physicist or machine learning scientist would write a math code in TensorFlow, PyTorch or JAX on a level that comfortable for them, and get as efficient code as possible by fully utilizing available resources, whether multiple GPUs or limited memory.

## Optimization approaches

TensorFlow, JAX and PyTorch build a static or a dynamic computational graph that consists of operations and tensors - operations produce tensors (data). Someone could describe this process as data flow through procedures in a graph. The resulting computational graph structure is known as a directed acyclic graph (DAG). Frameworks execute the graph using a stream executer (TensorFlow terminology), which in turn either runs operations on CPU or GPU depending on availability. Alternatively, we could optimise the computational graph via changing DAG traverse order, reshuffling operations (nodes) and by deciding on which device and what's more important - how to execute them on multiple devices (GPU).

Showing real examples, we propose three approaches for optimising computational graphs which could be integrated into MLIR (XLA):

### Evaluation order

Let consider a simple math expression that involves single matrix-matrix multiplication and matrix-vector multiplication. For a given N-by-M matrix $A$, M-by-D matrix $B$, and D-by-1 vector $v$, we have:

$$
c =  (A \times B) \times v
$$

Resulting vector $c$ has dimensionality N-by-1. The result in the parenthesis equals to N-by-D matrix, and the cost of that operation is $\mathcal{O}(NMD)$. Following vector multiplication costs another $\mathcal{O}(ND)$ and gives N-by-1 $c$ vector.

This order of execution is neither memory nor CPU/GPU clock efficient. A better choice would be to traverse the graph a bit differently:

$$
c =  A \times (B \times v)
$$

$B \times v$ matrix-vector multiplication costs $\mathcal{O}(MD)$ with a temporary $\mathcal{O}(M)$ memory footprint. Next in turn is again the vector-matrix multiplication with $\mathcal{O}(NM)$ cost.

The conclusion is:

- Perform vector multiplication first - it is always cheaper
- By changing the order of matrix operations, we can speed up algorithms and save some memory.

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

TensorFlow and PyTorch offer evaluations on GPU and CPU devices with fully materialized tensors. If a program cannot allocate memory for a tensor on a device, it will crash with OOM error. A user could prevent the OOM behaviour by splitting arrays into slices (blocks), treating these slices independently, and evaluating operations in a lazy and distributed manner, engaging all available devices. Also, if an operation cannot be applied to all slices at once, a user can decide to cache slices and run the operation sequentially on a subset. Of course, the latter approach might run slower, although the benefit is that the user would be feasible to run that code even under hard constraints.

Matrix multiplication is a perfect example for a map-reduce scheme. For a given matrix $A$, N-by-D size, and a matrix $B$, D-by-M size, matrix multiplication as mentioned earlier costs $\mathcal{O}(NDM)$ and because each element $C_{i,j}$ of the output matrix $C = A \times B$ is independent of other elements, this operation is highly parallelizable. GPU-accelerated libraries CUDA and MAGMA have super-efficient implementation for this op. The pitfall is in a GPU memory limitation. For encountered matrices with large N, D or M, temporary computations will not fit into a GPU memory.

...

## Solutions

The prerequisite for proficiency in performance techniques like multi-threading, caching, and distributed computing puts a strain on user experience and shifts scientists' focus from writing clean math code to an area outside their purview.

There are extensions to PyTorch that support map-reduce (and caching) and make life much easier, like [TensorComprehensions][2] and [KeOps][1]. To my knowledge, KeOps is the most successful and powerful tool that leverages the conception of lazy tensors, symbolic graph representations and JIT compilation. In short, KeOps tracks down calculations with lazy tensors and builds symbolic representation. When a program calls for a materialized result, KeOps generates and compiles efficient map-reduce code for a given expression that can be cached and run on multiple GPUs. Although a disadvantage is that users still have to define lazy tensors manually, and a framework must support tensor redefinition through either inheritance or traits. E.g. PyTorch supports tensor overriding, and TensorFlow doesn't. This leaves lots of TensorFlow users and their code behind.

[XLA][5] is a compiler for linear algebra and all listed frameworks - PyTorch, TensorFlow and JAX support it. This is an obvious choice for making tweaking of user-defined expressions (computational graphs). It can be done implicitly without user interventions.

**TO BE CONTINUED...**

## Notes on building TFL/ XLA

1. Follow the steps at https://www.tensorflow.org/install/source?hl=en#docker_linux_builds (use docker on linux, otherwise the build will take forever, since docker on MacOS is running in a VM; note that the build will take around 2-5 hours)

2. Make sure to set the bazel cache directory to within the mounted files, so they are not lost when you restart your contaier (VERY IMPORTANT, unless you like waiting for 5h while bazel fries your CPU) To do this run bazel with the following options:

```bash
# E.g. building the pip package
bazel --output_user_root=/mnt/.cache/bazel/_bazel_root/ build //tensorflow/tools/pip_package:build_pip_package
# E.g. running XLA tests
bazel --output_user_root=/mnt/.cache/bazel/_bazel_root/ test //tensorflow/compiler/xla/...
```

3. If you didn't add the flags on the first run, you can get the cache directory to the correct place by running `cp /root/.cache /mnt/.cache` (TFL is huge, so this can take a while ...)

4. Actually, using the path flag can cause issue, so do this: `ln -s /mnt/.cache /root/.cache` ;)

## References

* [KeOps framework][1]
* [TensorComprehensions framework][2]
* [KeOps design choices][3]
* [FalconML library powered by KeOps][4]
* [XLA][5]
* [MLIR presentation][6]
* [Cannon’s algorithm for distributed matrix multiplication][8]


[1]: https://www.kernel-operations.io/keops/index.html "KeOps framework"
[2]: https://facebookresearch.github.io/TensorComprehensions "TensorComprehensions"
[3]: https://www.kernel-operations.io/keops/formulas/design_choices.html "KeOps design choices"
[4]: https://twitter.com/luigicarratino/status/1313879062075539457?s=19 "FalconML library"
[5]: https://www.tensorflow.org/xla "XLA"
[6]: https://www.youtube.com/watch?v=qzljG6DKgic "MLIR presentation"
[7]: http://www.netlib.org/lapack/lawnspdf/lawn129.pdf "Parallel Matrix Multiplication Algorithm on Distributed-Memory Concurrent Computers"
[8]: https://en.wikipedia.org/wiki/Cannon%27s_algorithm "Cannon’s algorithm for distributed matrix multiplication"
[9]: https://blog.tensorflow.org/2019/04/mlir-new-intermediate-representation.html "MLIR"


