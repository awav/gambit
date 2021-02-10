Machine learning frameworks with automatic differentiation such as Theano, Caffe, TensorFlow, JAX, and PyTorch transformed and changed the field's pace. The crucial part of the success was utilizing modern hardware resources like GPU abstracting away the complexity of underlying architectures [1]. These frameworks allowed researchers and engineers to focus on problems and not their realizations. In turn, the hardware has limitations and imposes other computational barriers such as hard memory constraints. Also, researchers often don't know tricks to improve the code's efficiency or simply prefer readability over speed. These obstacles can be addressed with a compiler, a program that turns the code from computationally inefficient to faster and less memory-hungry representation. In this project, you will work on linear algebra optimizations for XLA compiler - the core of the JAX and a vital feature of the TensorFlow and PyTorch frameworks. The result of this project will impact a diverse set of science topics: kernel methods, Gaussian processes, optimal transport, applications in physics, etc.
You will learn:
- How to make linear algebra operations computationally and memory-efficient (apply Woodbury formula, using matrix properties) [2]
- How to run linear algebra operations in distributed fashion [3, 4]
- About computational graphs and their optimizations
- About compilers and XLA in particular [5]
The outcome of the project:
- Develop an XLA optimization routine for automatic detection where in an algorithm, it is possible to achieve a minimal memory footprint and potential computational benefits by applying the reshuffling of a sequence of operations.
- Set the ground for map-reduce based operations to execute matrix multiplications on super big matrices (N^2 and N > 1e5) in XLA.

Requirement: knowledge of C++, compilers.

[1] Rich Sutton's "The Bitter Lesson", http://www.incompleteideas.net/IncIdeas/BitterLesson.html \
[2] Automatic Generation of Efficient Linear Algebra, https://arxiv.org/pdf/1912.12924.pdf \
[3] KeOps, https://github.com/getkeops/keops \
[4] Cannon algorithm, https://en.wikipedia.org/wiki/Cannon%27s_algorithm \
[5] XLA website, https://www.tensorflow.org/xla \
[6] https://mlir.llvm.org/docs/Dialects/Linalg/
