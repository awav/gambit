* SGPR HLO IR test / graph plotting

TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES="3" DUMPDIR="xla-spgr-hlo-ir" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_tensor_size_threshold=1GB --xla_tensor_split_size=1GB --xla_enable_hlo_passes_only=tensor-splitter,algebraic-rewriter,dce,broadcast-simplifier,cholesky_expander,triangular_solve_expander,bitcast_dtypes_expander,CallInliner,gpu_scatter_expander,rce-optimizer" python ./sgpr_for_hlo_ir.py -l logs/sgpr_hlo_ir_1GB -s 777 -d 3droad -c xla -m 3000 2>&1 | tee logs-sgpr-hlo-ir.log

TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES="3" DUMPDIR="xla-spgr-hlo-ir" XLA_FLAGS="--xla_tensor_size_threshold=1GB --xla_tensor_split_size=1GB --xla_disable_hlo_passes=tensor-splitter" python ./sgpr_for_hlo_ir.py -l logs/sgpr_hlo_ir_1GB -s 777 -d 3droad -c xla -m 3000 2>&1 | tee "$DUMPDIR/std.log"
