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


with h5py.File("CE_Bavera_2020.h5", "r") as hf:
  np.random.seed(69)
  theta_train = hf["train_theta"][()]
  N = len(theta_train)
  rand_indices = np.random.choice(N, size = int(N/500))
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
  theta[:,0 ] = np.log((m1*m2)**(3/5)/((m1+m2)**(1/5)))
  theta[:, 1] = theta_true[:,-2]

  
  theta_true = theta.copy()
  theta = [ ]
  
  test_lambda = hf["test_lambda"][()]

print(thetas.shape, lambdas.shape, theta_true.shape, test_lambda.shape)

key = random.PRNGKey(0)
hidden_dims = [300, 150, 150]
num_flows = 6

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

masks, param_shapes, mask_skips, permutations = [], [], [], []
for _ in range(num_flows):
    param_shapes.append(param_shape)
    mask, mask_skip, perm = mask_generator()
    masks.append(mask)
    permutations.append(perm)
    mask_skips.append(mask_skip)


bounds = None



################
# Train HMC ####
################




with open('flow_params.pkl', "rb") as pf:
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
    
    
        

fig.savefig("prod_data_results/test_McChi_new_bounded.png")

sm = 0.35

flow = make_normalizing_flow(transform, theta_train, masks, mask_skips, permutations, bounds = bounds, context = lambda_train)
model, guide, guided_model, unravel_fn = bayesian_normalizing_flow(flow["lp"], best_params, scale_max = sm, multi_scale = False)#, scale_max = 0.1)


posterior_samples = train_bayesian_flow_hmc(model, unravel_fn, scale_max = sm, num_warmup = 1000, num_samples = 1000, target_accept = 0.8, num_chains = 4)#, anealing = False)#True)


with open("prod_data_results/bayesian_flow_samples_2.pkl", "wb") as pf:
    pickle.dump(posterior_samples, pf)

prior_samples = train_bayesian_flow_prior(model, unravel_fn, scale_max=sm, num_samples = 1000*4)

with open("prod_data_results/bayesian_flow_prior_samples_2.pkl", "wb") as pf:
    pickle.dump(prior_samples, pf)

with open("prod_data_results/bayesian_flow_samples_2.pkl", "rb") as pf:
    posterior_samples = pickle.load(pf)

with open("prod_data_results/bayesian_flow_prior_samples_2.pkl", "rb") as pf:
    prior_samples = pickle.load(pf)


ns = len(posterior_samples["params"][0][0][0])




def hpd(samples,alpha = 0.1):
    x=np.sort(np.copy(samples))
    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    
    return [hdi_min, hdi_max]


ranges = [(theta_true[:,i].min(), theta_true[:,i].max()) for i in range(theta_true.shape[-1])]
import tqdm
M1 = np.linspace(ranges[0][0], ranges[0][1], 100)
M2 = np.linspace(ranges[1][0], ranges[1][1], 100)
m1, m2 = np.meshgrid(M1,M2)
m1m2 = jnp.array([m1.flatten(),m2.flatten()]).T

flow_plotter = make_normalizing_flow(transform, m1m2, masks, mask_skips, permutations, bounds = bounds, context = test_lambda)

pdfs_post = [ ]
for j in tqdm.tqdm(range(ns)):
    i = j#indices[j]
    this_p = [[(this_rp[0][i], this_rp[1][i]) for this_rp in rp] for rp in posterior_samples["params"]]
    this_pdf = np.array(flow_plotter["lp"](this_p))
    pdfs_post.append(this_pdf.reshape(100,100))
    


m1m2_samples = [ ]
import tqdm
pdfs_prior = [ ]
for i in tqdm.tqdm(range(ns)):
    this_p = [[(this_rp[0][i], this_rp[1][i]) for this_rp in rp] for rp in prior_samples["params"]]
    this_pdf = np.array(flow_plotter["lp"](this_p))
    pdfs_prior.append(this_pdf.reshape(100,100))

fig , ax = plt.subplots(2, dpi = 100, tight_layout = True, figsize=(9*1.3,6*1.3*2))
labels = [r"$\log\mathcal{M}$", r"$\chi_{eff}$"]
ylims = [1.2, 25.5]
for i in range(theta_true.shape[-1]):
    
    this_range = [M1,M2][i]
    other_range = [M2,M1][i]
    
    pdf_prior = []
    for this_pdf in tqdm.tqdm(pdfs_prior):
        this_1d_pdf = np.trapz(np.exp(this_pdf), other_range, axis = i)
        pdf_prior.append(this_1d_pdf/np.trapz(this_1d_pdf,this_range))
    pdf_prior = np.array(pdf_prior)
    pdf_posterior = [ ]
    for k,this_pdf in enumerate(tqdm.tqdm(pdfs_post)):
        this_1d_pdf = np.trapz(np.exp(this_pdf-this_pdf.max()), other_range, axis = i)
        
        pdf_posterior.append(this_1d_pdf/np.trapz(this_1d_pdf,this_range))
        
    pdf_posterior = np.array(pdf_posterior)
    
    quantiles = np.array([hpd(this_pdf) for this_pdf in pdf_posterior.T])
    ax[i].fill_between(this_range, quantiles[:,0], quantiles[:,1],
                         alpha = 0.2, color = colors[0],
                         label = "Posterior")
    quantiles = np.array([hpd(this_pdf) for this_pdf in pdf_prior.T])
    ax[i].plot(this_range, quantiles[:,0], '--', color = colors[2], label = "Prior")
    ax[i].plot(this_range, quantiles[:,1], '--', color = colors[2])
    
    ax[i].hist(theta_true[:,i], histtype="step", color = colors[1], linewidth = 2, label = "True", bins = this_range, density = True)
    ax[i].set_xlabel(labels[i], fontsize = 32)
    ax[i].legend(fontsize = 25, loc = "upper right")
    ax[i].set_ylim(0.1,ylims[i])
    ax[i].set_yscale("log")
fig.tight_layout()
plt.show()
fig.savefig("prod_data_results/hmc_2param_2.png")

def find_level(density, mass=0.9):
    sorted_density = np.sort(density.ravel())[::-1]
    cumsum = np.cumsum(sorted_density)
    cumsum /= cumsum[-1]
    return sorted_density[np.searchsorted(cumsum, mass)]


#fig , ax = plt.subplots(1, dpi = 100, tight_layout = True, figsize=(9*1.3,6*1.3))
