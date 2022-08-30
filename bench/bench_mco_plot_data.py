import sys
import json
from pathlib import Path
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
import gpflow
import os

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from clitypes import LogdirPath
from bench_utils import BenchRunner, store_dict_as_h5
from bench_sgpr_utils import (
    compile_function,
    CompileType
)

__default_gambit_logs = "./logs_mco_default"
__gpu_devices = tf.config.get_visible_devices("GPU")
__gpu_dev = __gpu_devices[0] if __gpu_devices else None


if __gpu_dev is not None:
    click.echo(f">>> GPU device information: {__gpu_dev}")
    click.echo(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(__gpu_dev, True)


@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("-m", "--memory-limit", type=str)
@click.option("-s", "--seed", type=int, default=2)
@click.option("-r", "--repeat", type=int, default=1)
@click.option("-w", "--warmup", type=int, default=1)
@click.option("-c", "--compile", default="xla", help="Compile function with xla, tf or none")
@click.option("-d", "--dump-name",  default="mco")
def main(
    memory_limit: int,
    logdir: str,
    seed: int,
    warmup: int,
    repeat: int,
    compile: Literal["xla", "tf", "none"],
    dump_name: Literal["mco", "none"],
):
    chain_length_list = [4,8,16,24,32,40,48,56,64,72]
    dump_file_path = __default_gambit_logs+"/mco_stat_"+dump_name+".txt"
    mco_stat = []
    for chain_length in chain_length_list:
        memory_limit = "none" if memory_limit is None else memory_limit
        seed = np.random.randint(0,100)
        info = {
            "chain_length": chain_length,
            "memory_limit": memory_limit,
            "seed": seed,
            "repeat": repeat,
            "warmup": warmup,
            "compile": compile,
        }
        info_str = json.dumps(info, indent=2)
        click.echo("===> Starting")
        click.echo(f"-> {info_str}")
        assert Path(logdir).exists()
        compile_flag: CompileType = compile if compile != "none" else None
        rng = np.random.RandomState(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        gpflow_dtype = gpflow.config.default_float()

        def ctt(x, dtype=None):
            dtype = gpflow_dtype if dtype is None else dtype
            return tf.convert_to_tensor(x, dtype=dtype)

        dim_file_name_prefix = __default_gambit_logs+"/mco_dimensions_"
        dim_file_path = dim_file_name_prefix + str(chain_length)+".txt"
        matrix_dims = np.array([])
        matrices =  []
        if not os.path.exists(dim_file_path):
            print(f"{dim_file_path} doesn't exist")
            matrix_dims = np.random.randint(1,5000,size=chain_length+1)
            np.savetxt(dim_file_path, matrix_dims, fmt='%d', delimiter=',')
        else:
            # generate dims according to chain_length
            print(f"{dim_file_path} exists")
            matrix_dims = np.loadtxt(dim_file_path, dtype=np.int32, delimiter=',')
        step_size=4
        for i in range(0,len(matrix_dims)-1):
            # matrices.append(tf.random.uniform(matrix_dims[i],matrix_dims[i+1]))
            A = tf.random.uniform((matrix_dims[i],matrix_dims[i+1]))
            # print(f'matrices[{i}] = {A}\n')
            matrices.append(A)
        def eval_test_mco(matrices):
            transpose=False
            for i in range(0,len(matrices),step=step_size):
                if transpose:
                    for j in range(min(i+step_size-1,len(matrices)-1),i-1,-1):
                        if j == min(i+step_size-1,len(matrices)-1):
                            tmp_result=tf.transpose(matrices[j])
                        else:
                            tmp_result=tmp_result@tf.transpose(matrices[j])
                else:
                    for j in range(i,min(i+step_size,len(matrices))):
                        if j == i:
                            tmp_result=matrices[j]
                        else:
                            tmp_result=tmp_result@matrices[j]
                if i==0:
                    res = tmp_result
                else:
                    if transpose:
                        res =res@tf.transpose(tmp_result)
                    else:
                        res =res@tmp_result
                transpose = not transpose    
            return res
        eval_test_mco_compiled = compile_function(eval_test_mco, compile_flag)

        bench_runner = BenchRunner(repeat=repeat, warmup=warmup, logdir=logdir)
        results = bench_runner.bench(eval_test_mco_compiled, [matrices])
        bench_table = {**info, **results}



        if "elapsed_stats" not in results or "mem_stats" not in results:
            click.echo("⚠️ No stats in the benchmark output ⚠️ ")
            raise click.exceptions.Exit(0)

        (elap_mu, elap_std) = results["elapsed_stats"]
        (mem_mu, mem_std) = results["mem_stats"]
        if __gpu_dev is not None:
            # turn into Mib
            mem_mu, mem_std = mem_mu/1024/1024, mem_std/1024/1024
        mco_stat.append([chain_length,elap_mu*1000,mem_mu])

        click.echo(
            "[Bench] Total stat, "
            f"chain_length={chain_length}: spent_avg={elap_mu}, spent_std={elap_std}, "
            f"mem_avg={mem_mu}, mem_std={mem_std}"
        )
    np.savetxt(dump_file_path, mco_stat, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()