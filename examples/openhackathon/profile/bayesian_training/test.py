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
import jax.numpy as jnp
import numpy as np
print(jax.devices())

@jax.jit
def func(x):
    return x @ x - x ** 0.5
func = jax.profiler.annotate_function(func, name="func_step")


z = jnp.array(np.random.uniform(0,1,(1000,1000)))
    
jax.profiler.start_trace("./reports/jax/test")
_ = func(z)
jax.block_until_ready(_)
for i in range(10):
    with jax.profiler.StepTraceAnnotation("func", step_num=i):
        z = func(z)
        jax.block_until_ready(z)

jax.profiler.stop_trace()

        

