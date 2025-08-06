import os 
import numpy as np 
import h5py
import pickle
import json

import jax.numpy as jnp
import jax
from jax import random
import tqdm
import argparse

from naz.flows.bflow_jax_maf import make_conditional_autoregressive_nn, make_masked_affine_autoregressive_transform, make_normalizing_flow, train_maf, bayesian_normalizing_flow, train_bayesian_flow_hmc, train_bayesian_flow_prior, train_bayesian_flow, torch_to_jax, compute_bic


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser=argparse.ArgumentParser()
parser.add_argument("--fthin", type = int)
parser.add_argument("--bflow", type = str)
parser.add_argument("--mle-flow", type = str)
args = parser.parse_args()

fthin = int(args.fthin)
mle_flow = args.mle_flow
bflow = args.bflow

os.environ["NPROC"]="1" 
os.environ["intra_op_parallelism_threads"]="1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


nc = 4
with h5py.File("__run__/CE_Bavera_2020.h5", "r") as hf:
  np.random.seed(69)
  theta_train = hf["train_theta"][()]
  N = len(theta_train)
  rand_indices = np.random.choice(N, size = int(N/fthin))
  thetas = theta_train[rand_indices,:]
  #thetas = np.zeros((len(theta_train),2))
  #m1, m2 = theta_train[:,0], theta_train[:, 1]
  #thetas[:, 0] = (m1*m2)**(3/5)/((m1+m2)**(1/5))
  #thetas[:, 1] = theta_train[:,-2]
  thetas[:,:2] = np.log(thetas[:,:2])
  theta_train = [ ]
  lambdas = hf["train_lambda"][()][rand_indices,:]
  theta_true = hf["test_theta"][()]
  
  theta = np.array(np.zeros((len(theta_true),2)))
  m1, m2 = theta_true[:,0], theta_true[:, 1]
  theta[:,0 ] = np.log( (m1*m2)**(3/5)/((m1+m2)**(1/5)))
  theta[:, 1] = theta_true[:,-2]

  
  theta_true = theta.copy()
  theta = [ ]
  
  test_lambda = hf["test_lambda"][()]

print(thetas.shape, lambdas.shape, theta_true.shape, test_lambda.shape)

key = random.PRNGKey(0)
hidden_dims = [150, 150, 150]

theta_train = thetas
lambda_train = lambdas
nn, param_shape, mask_generator = make_conditional_autoregressive_nn(theta_train.shape[-1], lambda_train.shape[-1], hidden_dims)
transform = make_masked_affine_autoregressive_transform(nn, thetas.shape[-1])
bounds = None




with open(f'__run__/{mle_flow}', "rb") as pf:
    model = pickle.load(pf)
    best_params, param_shapes, masks, mask_skips, permutations= torch_to_jax(model)




flow = make_normalizing_flow(transform, theta_train, masks, mask_skips, permutations, bounds = bounds, context = lambda_train)


with open(f"__run__/{bflow}", "rb") as pf:
    posterior_samples = pickle.load(pf)

if "checkpoint" not in bflow:
    print(posterior_samples.keys())
    complexity = posterior_samples["standart_params"].shape[-1]
    print(complexity)
    ns = len(posterior_samples["params"][0][0][0])
else:
    _,_,_, unravel_fn = bayesian_normalizing_flow(flow["lp"], best_params, scale_max = 0.25, multi_scale = False, avg = False)#, scale_max = 0.1)
    posterior_samples["params"] = jax.jit(jax.vmap(unravel_fn))(posterior_samples["params"])
    ns = len(posterior_samples["params"][0][0][0])

    print(posterior_samples.keys())
    complexity = posterior_samples["standart_params"].shape[-1]
    print(complexity)
    
ns = len(posterior_samples["params"][0][0][0])
log_ls =  [ ]
for i in tqdm.tqdm(range(ns)):
    this_p = [[(this_rp[0][i], this_rp[1][i]) for this_rp in rp] for rp in posterior_samples["params"]]
    log_ls.append(flow["lp"](this_p).sum())
    print(log_ls[i])

bic = compute_bic(jnp.array(log_ls), len(thetas), complexity)
#bic = complexity * jnp.log(len(thetas)) - 2.* max_logl
print(bflow, f"complexity: {complexity}, dataset size: {len(thetas)},  BIC: {bic}, max log_l: {max(log_ls)}")

with open(f"__run__/{bflow[:-4]}_bic.json", "w") as jf:
    data = {"complexity":complexity, "dataset size": len(thetas), "BIC": np.float64(bic), "log_l_max": max(log_ls).item()  }
    for key, val in data.items():
        print(key, type(val), val)
    
    json.dump(data, jf)


