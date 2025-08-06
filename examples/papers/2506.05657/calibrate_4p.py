import os 
import numpy as np 
import h5py
import pickle
import glob

import jax.numpy as jnp
import jax
from jax import random
import tqdm

import matplotlib
from matplotlib.transforms import Bbox
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.lines import Line2D

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
colors=sns.color_palette('colorblind')
fs=28
import sys
plot_dir = 'plots/'

import argparse

print("blah")

from naz.flows.bflow_jax_maf import make_conditional_autoregressive_nn, make_masked_affine_autoregressive_transform, make_normalizing_flow, train_maf, bayesian_normalizing_flow, train_bayesian_flow_hmc, train_bayesian_flow_prior, train_bayesian_flow, torch_to_jax, calibrate



from naz.statutils import hpd, hpd_vectorized


###############
# Train MLE ###
###############

os.environ["NPROC"]="1" 
os.environ["intra_op_parallelism_threads"]="1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser=argparse.ArgumentParser()
parser.add_argument("--posterior-samples", type = str)
parser.add_argument("--prior-samples", type = str)
parser.add_argument("--rerun-dir", type = str)
parser.add_argument("--mle-flow", type = str)
parser.add_argument("--label", type = str)
args = parser.parse_args()


mle_flow = args.mle_flow
bflow_post = args.posterior_samples
bflow_prior = args.prior_samples
rerun_dir = args.rerun_dir
label = args.label




key = random.PRNGKey(0)
hidden_dims = [150, 150, 150]

bounds = None


with h5py.File("__run__/CE_Bavera_2020.h5", "r") as hf:
  theta_train = hf["train_theta"][()]
  thetas = theta_train.copy() 
  thetas[:,:2] = np.log(thetas[:,:2])
  theta_true = hf["test_theta"][()]
  
  theta_true = np.loadtxt('dense_test_set_in_range.txt') 
  test_lambda = hf["test_lambda"][()]

theta_true[:,:2] = np.log(theta_true[:,:2])
nn, param_shape, mask_generator = make_conditional_autoregressive_nn(theta_true.shape[-1], test_lambda.shape[-1], hidden_dims)
transform = make_masked_affine_autoregressive_transform(nn, theta_true.shape[-1])


with open(f'__run__/{mle_flow}', "rb") as pf:
    model = pickle.load(pf)
    best_params, param_shapes, masks, mask_skips, permutations= torch_to_jax(model)


ranges = [[theta_true[:,i].min(), theta_true[:,i].max()] for i in range(theta_true.shape[-1])]

ranges[0][0]=0.8#1.2#0.8
ranges[0][1]=4.5#1.2
ranges[1][0]=0.8#0.8
ranges[1][1]=4.5#1.2
ranges[2][0]=0.05
ranges[2][1]=0.7
ranges[3][0]=0
ranges[3][1]=8
print(ranges)
print(theta_true[:,-1])
flow_plotter = make_normalizing_flow(transform, theta_true, masks, mask_skips, permutations, bounds = bounds, context = test_lambda)
#print(f"ppds_pdf_{label}")
if not os.path.exists(f"ppd_cal_pdfs_{label}.pkl"):

    with open(f"__run__/{bflow_post}", "rb") as pf:
        posterior_samples = pickle.load(pf)

    if "checkpoint" not in bflow_post:
        ns = len(posterior_samples["params"][0][0][0])
    else:
        _,_,_, unravel_fn = bayesian_normalizing_flow(flow_plotter["lp"], best_params, scale_max = 0.25, multi_scale = False, avg = False)#, scale_max = 0.1)
        posterior_samples["params"] = jax.jit(jax.vmap(unravel_fn))(posterior_samples["params"])
        ns = len(posterior_samples["params"][0][0][0])
    print(f"Number of posterior samples: {ns}")




    ppds = [ ]
    nsamp = 800000
    keys = jax.random.split(jax.random.PRNGKey(0), ns)
    for i in tqdm.tqdm(range(ns)):
        this_p = [[(this_rp[0][i], this_rp[1][i]) for this_rp in rp] for rp in posterior_samples["params"]]
        this_pdf = np.array(flow_plotter["sampler"](this_p, keys[i], nsamp)[0])
        ppds.append(this_pdf)
    


    #with h5py.File(f"ppd_cal_pdfs_{label}.h5", "w") as hf:
    #    hf.create_dataset('post', data = jnp.array(ppds))
    with open(f"ppd_cal_pdfs_{label}.pkl", "wb") as pf:
        pickle.dump(ppds, pf)
else:
    #with h5py.File(f"ppd_cal_pdfs_{label}.h5", "r") as hf:
    #    ppds = hf["post"][()]
    with open(f"ppd_cal_pdfs_{label}.pkl", "rb") as pf:
        ppds = pickle.load(pf)


nQ = np.array([5**4, 7**4, 10**4, 20**4])#np.arange(11)[1:]**2
#nQ = np.array([30**4, 40**4, 50**4])
cs = np.linspace(0.1,0.95,10)
plt.figure()
ec = [ ]
for nq in nQ:
    coverage = calibrate(np.array(ppds), theta_true, nq, cs, fthin = 100, itype='eqt', twod = False, ranges = ranges)
    print(nq)
    plt.plot(cs, coverage, label = f'nQ={nq}')
    ec.append(coverage)

plt.plot(cs,cs, color = 'black')

plt.legend()
plt.savefig(f'__run__/calibration_{label}.png')
    

ec = np.array(ec)
np.savetxt(f'__run__/{label}_coverage.txt', np.array(ec.T))




