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

from naz.flows.bflow_jax_maf import make_conditional_autoregressive_nn, make_masked_affine_autoregressive_transform, make_normalizing_flow, train_maf, bayesian_normalizing_flow, train_bayesian_flow_hmc, train_bayesian_flow_prior, train_bayesian_flow, torch_to_jax 


from naz.statutils import hpd, hpd_vectorized, find_level

from physt import h2

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


with h5py.File("__run__/CE_Bavera_2020.h5", "r") as hf:
  
  theta_true = hf["test_theta"][()]
  
  theta = np.zeros((len(theta_true),2))
  m1, m2 = theta_true[:,0], theta_true[:, 1]
  theta[:,0 ] = np.log( (m1*m2)**(3/5)/((m1+m2)**(1/5)))
  theta[:, 1] = theta_true[:,-2]

  
  theta_true = theta.copy()
  theta = [ ]
  
  test_lambda = hf["test_lambda"][()]


key = random.PRNGKey(0)
hidden_dims = [150, 150, 150]

nn, param_shape, mask_generator = make_conditional_autoregressive_nn(theta_true.shape[-1], test_lambda.shape[-1], hidden_dims)
transform = make_masked_affine_autoregressive_transform(nn, theta_true.shape[-1])

bounds = None





with open(f'__run__/{mle_flow}', "rb") as pf:
    model = pickle.load(pf)
    best_params, param_shapes, masks, mask_skips, permutations= torch_to_jax(model)
print(param_shapes)
nbins1 = 15+1
nbins2 = 30+1
ranges = [[theta_true[:,i].min(), theta_true[:,i].max()] for i in range(theta_true.shape[-1])]

ranges[0][0]*=0.8
ranges[0][1]*=1.2

M1 = np.linspace(ranges[0][0], ranges[0][1], 100)
M2 = np.linspace(ranges[1][0], ranges[1][1], 100)
_M1 = np.linspace(ranges[0][0], ranges[0][1], nbins1)
_M2 = np.linspace(ranges[1][0], ranges[1][1], nbins2)
_M2 = np.sort(np.append(_M2[_M2>=0.12], M2[M2<=0.12]))


ngrid1 = len(M1)


ngrid2 = len(M2)
m1, m2 = np.meshgrid(M1,M2)


m1m2 = jnp.array([m1.flatten(),m2.flatten()]).T
flow_plotter = make_normalizing_flow(transform,m1m2, masks, mask_skips, permutations, bounds = bounds, context = test_lambda)
pdf = np.asarray(flow_plotter["lp"](best_params)).reshape(ngrid2,ngrid1)


with open(f"__run__/{bflow_post}", "rb") as pf:
    posterior_samples = pickle.load(pf)

with open(f"__run__/{bflow_prior}", "rb") as pf:
    prior_samples = pickle.load(pf)

if "checkpoint" not in bflow_post:
    ns = len(posterior_samples["params"][0][0][0])
else:
    _,_,_, unravel_fn = bayesian_normalizing_flow(flow_plotter["lp"], best_params, scale_max = 0.25, multi_scale = False, avg = False)#, scale_max = 0.1)
    posterior_samples["params"] = jax.jit(jax.vmap(unravel_fn))(posterior_samples["params"])
    ns = len(posterior_samples["params"][0][0][0])
print(f"Number of posterior samples: {ns}")

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

def hpd_vectorized(samples, alpha=0.1):
    """
    Vectorized HPD for samples with shape (ns, nx, ny), returns (2, nx, ny)
    """
    # Sort along the sample axis (axis=0)
    x = np.sort(samples, axis=0)
    ns = x.shape[0]
    cred_mass = 1.0 - alpha
    interval_idx_inc = int(np.floor(cred_mass * ns))
    n_intervals = ns - interval_idx_inc

    if n_intervals <= 0:
        raise ValueError("Too few elements for interval calculation")

    # Compute interval widths: shape (n_intervals, nx, ny)
    interval_width = x[interval_idx_inc:, ...] - x[:n_intervals, ...]

    # Find the index with the smallest width along axis 0
    min_idx = np.argmin(interval_width, axis=0)  # shape (nx, ny)

    # Prepare an output array of shape (2, nx, ny)
    hdi_min = np.take_along_axis(x, min_idx[None, :, :], axis=0)[0]
    hdi_max = np.take_along_axis(x, (min_idx + interval_idx_inc)[None, :, :], axis=0)[0]

    return np.stack([hdi_min, hdi_max], axis=0)


flow_plotter = make_normalizing_flow(transform, m1m2, masks, mask_skips, permutations, bounds = bounds, context = test_lambda)
'''
pdfs_post = [ ]
for i in tqdm.tqdm(range(ns)):
    this_p = [[(this_rp[0][i], this_rp[1][i]) for this_rp in rp] for rp in posterior_samples["params"]]
    this_pdf = np.array(flow_plotter["lp"](this_p))
    pdfs_post.append(this_pdf.reshape(ngrid2,ngrid1))
    



pdfs_prior = [ ]
for i in tqdm.tqdm(range(ns)):
    this_p = [[(this_rp[0][i], this_rp[1][i]) for this_rp in rp] for rp in prior_samples["params"]]
    this_pdf = np.array(flow_plotter["lp"](this_p))
    pdfs_prior.append(this_pdf.reshape(ngrid2,ngrid1))

with h5py.File(f"ppd_pdfs_{label}.h5", "w") as hf:
    hf.create_dataset("post", data = np.array(pdfs_post))
    hf.create_dataset("prior", data = np.array(pdfs_prior))

'''

with h5py.File(f"ppd_pdfs_{label}.h5", "r") as hf:
    pdfs_post = hf["post"][()]
    pdfs_prior = hf["prior"][()]
mle_flows = glob.glob(f"__run__/{rerun_dir}/*.pkl")
print(len(mle_flows))
pdfs_mle = [ ]
for mle_flow in tqdm.tqdm(mle_flows):
    with open(mle_flow, "rb") as pf:
        model = pickle.load(pf)
    this_best_param, param_shapes, masks, mask_skips, permutations= torch_to_jax(model)
    this_flow_plotter = make_normalizing_flow(transform, m1m2, masks, mask_skips, permutations, bounds = bounds, context = test_lambda)
    this_pdf = np.array(this_flow_plotter["lp"](this_best_param))
    pdfs_mle.append(this_pdf.reshape(ngrid2,ngrid1))
fig , ax = plt.subplots(1,3, dpi = 500, tight_layout = True, figsize=(9*1.3*3,6*1.3))
labels = [r"$\log\frac{\mathcal{M}}{M_{\odot}}$", r"$\chi_{eff}$"]

ylims = [(0.01, 2), (0.1,45.5)]
for i in range(theta_true.shape[-1]):
    
    this_range = [M1,M2][i]
    this_bin = [_M1, _M2][i]
    other_range = [M2,M1][i]
    this_axis = i
    pdf_prior = []
    for this_pdf in tqdm.tqdm(pdfs_prior):
        
        this_1d_pdf = np.trapz(np.exp(this_pdf), other_range, axis = this_axis)
        pdf_prior.append(this_1d_pdf/np.trapz(this_1d_pdf,this_range))
    pdf_prior = np.array(pdf_prior)
    pdf_posterior = [ ]
    for k,this_pdf in enumerate(tqdm.tqdm(pdfs_post)):
        this_1d_pdf = np.trapz(np.exp(this_pdf-this_pdf.max()), other_range, axis = this_axis)
        
        pdf_posterior.append(this_1d_pdf/np.trapz(this_1d_pdf,this_range))
        
    pdf_posterior = np.array(pdf_posterior)

    
    
        
    
    
    quantiles = np.array([hpd(this_pdf) for this_pdf in pdf_posterior.T])
    ax[i].fill_between(this_range, quantiles[:,0], quantiles[:,1],
                         alpha = 0.2, color = colors[0],
                         label = "Posterior")
    quantiles = np.array([hpd(this_pdf) for this_pdf in pdf_prior.T])
    ax[i].plot(this_range, quantiles[:,0], '--', color = colors[2], label = "Prior")
    ax[i].plot(this_range, quantiles[:,1], '--', color = colors[2])
            
    
    ax[i].hist(theta_true[:,i], histtype="step", color = colors[1], linewidth = 2, label = "True", bins = this_bin, density = True)
    ax[i].set_xlabel(labels[i], fontsize = 32)
    #if i == 0:
    ax[i].legend(fontsize = 25, loc = "upper right")
    ax[i].set_ylim(*ylims[i])
    ax[i].set_yscale("log")



twod_pdfs = np.array(pdfs_post)
twod_pdfs = np.exp(twod_pdfs-twod_pdfs.max())
twod_pdfs/= np.trapz(np.trapz(twod_pdfs, M2,axis=1), M1, axis=1)[:,None,None]
twod_pdfs = np.mean(twod_pdfs, axis = 0)
print(twod_pdfs.shape)
level = find_level(twod_pdfs, 0.9)




cs0 = ax[-1].contour(m1, m2, twod_pdfs, levels=[level], colors=[colors[0]], linewidths=2.0)
#cs.collections[0].set_label("Posterior predictive")


twod_pdfs = np.array(pdfs_prior)
twod_pdfs = np.exp(twod_pdfs-twod_pdfs.max())
twod_pdfs/= np.trapz(np.trapz(twod_pdfs, M2,axis=1), M1, axis=1)[:,None,None]
twod_pdfs = np.mean(twod_pdfs, axis = 0)
print(twod_pdfs.shape)
level = find_level(twod_pdfs, 0.9)




cs1 = ax[-1].contour(m1, m2, twod_pdfs, levels=[level], colors=[colors[2]], linewidths=2.0)
#cs.collections[0].set_label("Prior predictive")



density,_,_ = np.histogram2d(theta_true[:,0], theta_true[:,1], bins = (_M1, _M2), density = True)
density = density.T
__M1 = (_M1[1:]+_M1[:-1])*0.5
__M2 = (_M2[1:]+_M2[:-1])*0.5
xx, yy = np.meshgrid(__M1, __M2)


level = find_level(density, 0.9)
cs2 = ax[-1].contour(xx,yy, density, [level], colors = [colors[1]], linewidth = 2.0)#, label = "Truth")
#cs2.collections[0].set_label("Truth")

for k,twod_pdf in enumerate(pdfs_mle):
    twod_pdfs = np.array(twod_pdf)
    twod_pdfs = np.exp(twod_pdfs-twod_pdfs.max())
    twod_pdfs/= np.trapz(np.trapz(twod_pdfs, M2,axis=0), M1)
    level = find_level(twod_pdfs, 0.9)
    if k == 0:
        print("cs3 exists")
        cs3 = ax[-1].contour(m1, m2, twod_pdfs, levels=[level], colors=[colors[4]], linewidths=0.6,alpha = 0.8)
        #cs3.collections[0]set_label("MLE reruns")
    else:
        ax[-1].contour(m1, m2, twod_pdfs, levels=[level], colors=[colors[4]], linewidths=0.6, alpha = 0.8)
ax[-1].set_xlabel(labels[0], fontsize = 32)
ax[-1].set_ylabel(labels[1], fontsize = 32)
#ax[-1].set_xlim()
custom_lines = [ ]
labels = ["Posterior predictive", "Prior predictive", "Truth", "MLE reruns"]
'''
for i,cs in enumerate([cs0, cs1, cs2, cs3]):
    contour_color = cs.collections[0].get_edgecolor()
    custom_lines.append(Line2D([0], [0], color=contour_color[0], lw=2, label=labels[i]))
ax[-1].legend(handles = custom_lines, fontsize = 25, loc="upper right", ncol=2)
'''
fig.tight_layout()
plt.show()
fig.savefig(f"__run__/hmc_{label}.pdf")


