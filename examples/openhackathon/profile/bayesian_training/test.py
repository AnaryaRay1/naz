import os
os.environ["NPROC"]="1"
os.environ["intra_op_parallelism_threads"]="1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import jax
print(jax.devices())
