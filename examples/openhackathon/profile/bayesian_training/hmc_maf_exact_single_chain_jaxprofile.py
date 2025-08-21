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

from naz.flows.bflow_jax_maf_jaxprofile import make_conditional_autoregressive_nn, make_masked_affine_autoregressive_transform, make_normalizing_flow, train_maf, bayesian_normalizing_flow, train_bayesian_flow_hmc, train_bayesian_flow_prior, train_bayesian_flow, torch_to_jax


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

'''
os.environ["NPROC"]="1" 
os.environ["intra_op_parallelism_threads"]="1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
'''
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
'''
def build_profiled_potential_fn(model, model_args, **model_kwargs):
    init_params, potential_fn_gen, postprocess_fn, model_trace = initialize_model(
        jax.random.PRNGKey(0),
        model,
        dynamic_args=True,
        model_args=model_args,
        model_kwargs=model_kwargs,
        forward_mode_differentiation=False,
    )

    # Get the actual potential function for this initialization
    potential_fn_raw = potential_fn_gen(init_params)   # callable: q -> U(q)

    # Compile, then label for profiler timelines
    potential_fn_jit = jax.jit(potential_fn_raw)
    profiled_potential_fn = jax.profiler.annotate_function(potential_fn_jit, name="potential_fn")

    return init_params, profiled_potential_fn, postprocess_fn
'''
nc = 1
with h5py.File("../../../../../../data/CE_Bavera_2020.h5", "r") as hf:
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

with open(f'{mle_flow}', "rb") as pf:
#    model = pickle.load(pf)
#    best_params, param_shapes, masks, mask_skips, permutations= torch_to_jax(model)
     data = pickle.load(pf)
best_params = data["params"]
masks = data["masks"]
mask_skips = data["mask_skips"]
permutations = data["permutations"]



out = f"{sm}_{fthin}_{nc}_{label}"


flow = make_normalizing_flow(transform, theta_train, masks, mask_skips, permutations, bounds = bounds, context = lambda_train)

model, guide, guided_model, unravel_fn = bayesian_normalizing_flow(flow["lp"], best_params, scale_max = sm, multi_scale = False)#, scale_max = 0.1)
print(sm, fthin)
'''
unravel_fn = jax.profiler.annotate_function(jax.jit(jax.vmap(unravel_fn)), name = "vec_unravel")
init_state, profiled_potential_fn, constrain_fn = build_profiled_potential_fn(model, (), scale_max = sm)
kernel = NUTS(potential_fn = profiled_potential_fn)
mcmc = MCMC(kernel, num_warmup=10, num_samples = 2, progress_bar = True)

mcmc.run(jax.random.PRNGKey(1), init_params=init_state.z)
sample = mcmc.get_samples()
_ = unravel_fn(sample["params"])

jax.profiler.start_trace("reports/jax/hmc_manual", create_perfetto_link=True)
with jax.profiler.annotate_function("mcmc_run_nuts"):
    mcmc.run(jax.random.PRNGKey(2))#, init_params=init_state.z)
    posterior = mcmc.get_samples()#group_by_chain=False)
    samples = unravel_fn(posterior["params"])
jax.profiler.stop_trace()


'''
#with jax.profiler.trace("reports/jax"):
A = jax.numpy.arange(1000000)
print(A**2)
chckpt = False
if not chckpt:
    jax.profiler.start_trace("reports/jax/hmc_manual")
    posterior_samples = train_bayesian_flow_hmc(model, unravel_fn, scale_max = sm, num_warmup = 10, num_samples = 10, target_accept = 0.8, num_chains = nc)#, anealing = False)#True)
    jax.profiler.stop_trace()
    import sys
    sys.exit()
else:
    posterior_samples = train_bayesian_flow(model, unravel_fn, scale_max = sm, num_warmup = nt, num_samples = ns, target_accept = 0.8, num_chains = nc, checkpoint_file = f"checkpoint_{out}.pkl", posterior_file = f"posterior_checkpoint_{out}_3.pkl", nbatch=100)#, anealing = False)#True)


with open(f"bayesian_flow_samples_{out}.pkl", "wb") as pf:
    pickle.dump(posterior_samples, pf)

prior_samples = train_bayesian_flow_prior(model, unravel_fn, scale_max=sm, num_samples = ns*nc)

with open(f"bayesian_flow_prior_samples_{out}.pkl", "wb") as pf:
    pickle.dump(prior_samples, pf)




