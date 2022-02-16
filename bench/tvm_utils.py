from typing import Optional
import tvm
import tensorflow as tf
import tvm.contrib.graph_executor as runtime
from tvm import relay
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def extract_tf_graph_def(func, *args, **kwargs):
    """
    Returns a graph definition of the tensorflow
    Args:
        *args and **kwargs:
    """
    concrete_func = func.get_concrete_function(*args, **kwargs)
    concrete_func_frozen = convert_variables_to_constants_v2(concrete_func)
    graph = concrete_func_frozen.graph
    graph_def = graph.as_graph_def(add_shapes=True)
    return graph_def


def tvm_compile_graph_executor(mod, params, target="llvm", target_host="llvm", opt_level=3):
    with tvm.transform.PassContext(opt_level):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
    return lib


def tvm_run_graph_executor(lib, inputs, ctx=tvm.cpu(0)):
    mod = runtime.GraphModule(lib["default"](ctx))
    [mod.set_input(i, v) for i, v in enumerate(inputs)]
    mod.run()
    return [mod.get_output(0).asnumpy()]


def run_graph_using_tvm(
    graph_def,
    inputs,
    output_tensors: Optional[tf.Tensor] = None,
    target: Optional[str] = "llvm",
    target_host: Optional[str] = "llvm",
    opt_level: int = 3,
    ctx=None,
):
    """
    TVM compile and run TF graph.
    Args:
        output_tensors: tf.Tensors. Remark: we cannot get this tensor for large models.
            Actually, we are interested in those models.
    """
    ctx = tvm.cpu(0) if ctx is None else ctx
    mod, params = relay.frontend.tensorflow.from_tensorflow(graph_def)
    lib = tvm_compile_graph_executor(
        mod, params, target=target, target_host=target_host, opt_level=opt_level
    )
    out = tvm_run_graph_executor(lib, inputs, ctx=ctx)
    return out
