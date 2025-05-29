import os 
import numpy as np 
import h5py
import pickle

import jax.numpy as jnp
import jax
from jax import random

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

from maf import make_conditional_autoregressive_nn, make_masked_affine_autoregressive_transform, make_normalizing_flow, train_maf, bayesian_normalizing_flow, train_bayesian_flow_hmc, train_bayesian_flow_prior, train_bayesian_flow

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
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

fthin = 500
nc = 4
with h5py.File("CE_Bavera_2020.h5", "r") as hf:
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
  print(np.unique(lambdas, axis = 0))
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

train_frac = 0.89
splitter = int(len(thetas)*train_frac)

indices = np.random.choice(len(thetas), size = len(thetas), replace = False)
theta_rand = thetas[indices,:]
lambda_rand = lambdas[indices,:]
theta_train = theta_rand[:splitter, :]
theta_test = theta_rand[splitter:, :]
lambda_train = lambda_rand[:splitter, :]
lambda_test = lambda_rand[splitter:, :]

nn, param_shape, mask_generator = make_conditional_autoregressive_nn(theta_train.shape[-1], lambda_train.shape[-1], hidden_dims)
transform = make_masked_affine_autoregressive_transform(nn, thetas.shape[-1])

bounds = None



################
# Train HMC ####
################


label = "150_3_16"

with open('prod_data_results/flow_params_mle_01_1_prod_2d_150_3_16_mcchi.pkl', "rb") as pf:
    data = pickle.load(pf)
best_params = data["params"]
masks = data["masks"]
mask_skips = data["mask_skips"]
permutations = data["permutations"]

ranges = [(theta_true[:,i].min(), theta_true[:,i].max()) for i in range(theta_true.shape[-1])]
M1 = np.linspace(ranges[0][0], ranges[0][1], 100)
M2 = np.linspace(ranges[1][0], ranges[1][1], 100)
m1, m2 = np.meshgrid(M1,M2)
m1m2 = jnp.array([m1.flatten(),m2.flatten()]).T
flow_plotter = make_normalizing_flow(transform,m1m2, masks, mask_skips, permutations, bounds = bounds, context = test_lambda)
pdf = np.asarray(flow_plotter["lp"](best_params)).reshape(100,100)




thetas_new, _ = flow_plotter["sampler"](best_params, jax.random.PRNGKey(0), 100000)
import corner
fig = corner.corner(np.array(theta_true), hist_kwargs = {"density":True}, color = colors[0], range = ranges, bins = 30+1, plot_datapoints = False)
axes = np.array(fig.axes).reshape(2,2)
for i in range(2):
    ax = axes[i,i]
    this_range = [M1,M2][i]
    other_range = [M2,M1][i]
    print(this_range.shape, pdf.shape)
    this_1d_pdf = np.trapz(np.exp(pdf), other_range, axis = i)
    this_1d_pdf/= np.trapz(this_1d_pdf,this_range)
    ax.plot(this_range,this_1d_pdf, color = colors[1])
    
    
        

fig.savefig(f"prod_data_results/test_McChi_new_bounded_{label}.png")

sm = 0.35

flow = make_normalizing_flow(transform, theta_train, masks, mask_skips, permutations, bounds = bounds, context = lambda_train)
model, guide, guided_model, unravel_fn = bayesian_normalizing_flow(flow["lp"], best_params, scale_max = sm, multi_scale = False)#, scale_max = 0.1)


posterior_samples = train_bayesian_flow_hmc(model, unravel_fn, scale_max = sm, num_warmup = 100, num_samples = 200, target_accept = 0.8, num_chains = nc)#, anealing = False)#True)


with open(f"prod_data_results/bayesian_flow_samples_{sm}_{fthin}_{nc}_{label}.pkl", "wb") as pf:
    pickle.dump(posterior_samples, pf)

prior_samples = train_bayesian_flow_prior(model, unravel_fn, scale_max=sm, num_samples = 1000*nc)

with open(f"prod_data_results/bayesian_flow_prior_samples_{sm}_{fthin}_{nc}_{label}.pkl", "wb") as pf:
    pickle.dump(prior_samples, pf)

