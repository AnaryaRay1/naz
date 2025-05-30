import os 
import numpy as np 
import h5py
import pickle

import jax.numpy as jnp
import jax
from jax import random
import argparse

import matplotlib
from matplotlib.transforms import Bbox
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.unicode_minus'] = False

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
colors=sns.color_palette('colorblind')
fs=28
import sys
plot_dir = 'plots/'

from naz.flows.bflow_jax_maf import make_conditional_autoregressive_nn, make_masked_affine_autoregressive_transform, make_normalizing_flow, train_maf, bayesian_normalizing_flow, train_bayesian_flow_hmc, train_bayesian_flow_prior, train_bayesian_flow, torch_to_jax


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
parser.add_argument("--num-warmup", type = int)
parser.add_argument("--num-samples", type = int)
parser.add_argument("--sigma", type = float)
parser.add_argument("--mle-flow", type = str)
parser.add_argument('--avg', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--chckpt', type=str2bool, nargs='?', const=True, default=False)
args = parser.parse_args()

fthin = int(args.fthin)
mle_flow = args.mle_flow
avg = args.avg
ns = args.num_samples
nt = args.num_warmup
sm = args.sigma
chckpt = args.chckpt

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
  theta_train = theta_train[rand_indices,:]
  thetas = np.zeros((len(theta_train),2))
  m1, m2 = theta_train[:,0], theta_train[:, 1]
  thetas[:, 0] = np.log((m1*m2)**(3/5)/((m1+m2)**(1/5)))
  thetas[:, 1] = theta_train[:,-2]
  theta_train = [ ]
  lambdas = hf["train_lambda"][()][rand_indices,:]
  theta_true = hf["test_theta"][()]
  
  theta = np.zeros((len(theta_true),2))
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



################
# Train HMC ####
################


label = f"150_3_16{'_avg' if avg else ''}"

with open(f'__run__/{mle_flow}', "rb") as pf:
    model = pickle.load(pf)
    best_params, param_shapes, masks, mask_skips, permutations= torch_to_jax(model)




out = f"{sm}_{fthin}_{nc}_{label}"

sm = 0.35

flow = make_normalizing_flow(transform, theta_train, masks, mask_skips, permutations, bounds = bounds, context = lambda_train)

model, guide, guided_model, unravel_fn = bayesian_normalizing_flow(flow["lp"], best_params, scale_max = sm, multi_scale = False, avg = avg)#, scale_max = 0.1)

if not chckpt:
    posterior_samples = train_bayesian_flow_hmc(model, unravel_fn, scale_max = sm, num_warmup = nt, num_samples = ns, target_accept = 0.8, num_chains = nc)#, anealing = False)#True)
else:
    posterior_samples = train_bayesian_flow(model, unravel_fn, scale_max = sm, num_warmup = nt, num_samples = ns, target_accept = 0.8, num_chains = nc, nbatch=100, checkpoint_file = f"__run__/checkpoint_{out}.pkl", posterior_file = f"__run__/posterior_checkpoint_{out}.pkl")#, anealing = False)#True)


with open(f"__run__/bayesian_flow_samples_{out}.pkl", "wb") as pf:
    pickle.dump(posterior_samples, pf)

prior_samples = train_bayesian_flow_prior(model, unravel_fn, scale_max=sm, num_samples = ns*nc)

with open(f"__run__/bayesian_flow_prior_samples_{out}.pkl", "wb") as pf:
    pickle.dump(prior_samples, pf)




