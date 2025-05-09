import os 
import numpy as np 
import h5py
import pickle

import jax.numpy as jnp
import jax
from jax import random

from maf import make_conditional_autoregressive_nn, make_masked_affine_autoregressive_transform, make_normalizing_flow, train_maf, bayesian_normalizing_flow, train_bayesian_flow_hmc, train_bayesian_flow_prior

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
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# os.environ["NPROC"]="1" 
# os.environ["intra_op_parallelism_threads"]="1" 
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
# os.environ["OPENBLAS_NUM_THREADS"]="1"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" 
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false" 
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" 
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


with h5py.File("CE_Bavera_2020.h5", "r") as hf:
  theta_train = hf["train_theta"][()]
  thetas = np.zeros((len(theta_train),2))
  m1, m2 = theta_train[:,0], theta_train[:, 1]
  thetas[:,0 ] = np.log((m1*m2)**(3/5)/((m1+m2)**(1/5)))
  thetas[:, 1] = theta_train[:,-2]
  theta_train = [ ]
  lambdas = hf["train_lambda"][()]
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
hidden_dims = [300, 150, 150, 150]
num_flows = 8

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

print(param_shapes)

bounds = None

flows = {"train":make_normalizing_flow(transform, theta_train, masks, mask_skips, permutations, bounds = bounds, context = lambda_train), "test": make_normalizing_flow(transform, theta_test, masks, mask_skips, permutations, bounds = bounds, context = lambda_test)}


params, train_params, train_loss, validation_loss = train_maf(flows["train"]["lp"], flows["test"]["lp"], param_shapes, lr = 1e-3, num_epochs = 2*14096, min_lr = 1e-8, patience = 128)

with open('flow_params_more_complex.pkl', 'wb') as pf:
    pickle.dump({"params": params, "train_params": train_params, "masks": masks, "mask_skips": mask_skips, "permutations": permutations, "theta_train": theta_train, "theta_test": theta_test, "bounds": bounds, "training_loss": train_loss, "validation_loss": validation_loss, "lambda_train":lambda_train, "lambda_test": lambda_test}, pf)



################
# Train HMC ####
################


with open('flow_params_more_complex.pkl', "rb") as pf:
    data = pickle.load(pf)
best_params = data["train_params"]
masks = data["masks"]
mask_skips = data["mask_skips"]
permutations = data["permutations"]
bounds = data["bounds"]
train_loss = data["training_loss"]
val_loss = data["validation_loss"]



ranges = [(theta_true[:,i].min(), theta_true[:,i].max()) for i in range(theta_true.shape[-1])]
flow_plotter = make_normalizing_flow(transform, theta_true, masks, mask_skips, permutations, bounds = bounds, context = test_lambda)
thetas_new, _ = flow_plotter["sampler"](best_params, jax.random.PRNGKey(0), 10000)
import corner
fig = corner.corner(np.array(theta_true), hist_kwargs = {"density":True}, color = "b", range = ranges, bins = 30+1)
fig = corner.corner(np.array(thetas_new), hist_kwargs = {"density":True}, color = "orange", range = ranges, bins = 30+1, fig = fig)
fig.savefig("test_McChi_new_bounded.png")


flow = make_normalizing_flow(transform, theta_train, masks, mask_skips, permutations, bounds = bounds, context = lambda_train)
model, guide, guided_model, unravel_fn = bayesian_normalizing_flow(flow["lp"], best_params, scale_max = 0.1, multi_scale = False)#, scale_max = 0.1)


posterior_samples = train_bayesian_flow_hmc(model, unravel_fn, scale_max=1.0, num_warmup = 100, num_samples = 700, target_accept = 0.8, num_chains = 1)


with open("bayesian_flow_samples_hmc_more_complex.pkl", "wb") as pf:
    pickle.dump(posterior_samples, pf)

prior_samples = train_bayesian_flow_prior(model, unravel_fn, scale_max=1.0, num_samples = 700*1)

with open("bayesian_flow_prior_samples_hmc_more_complex.pkl", "wb") as pf:
    pickle.dump(prior_samples, pf)

with open("bayesian_flow_samples_hmc_more_complex.pkl", "rb") as pf:
    posterior_samples = pickle.load(pf)
ns = len(posterior_samples["params"][0][0][0])


m1m2_samples = [ ]
import tqdm
flow_plotter = make_normalizing_flow(transform, theta_true, masks, mask_skips, permutations, bounds = bounds, context = test_lambda)
for i in tqdm.tqdm(range(ns)):
    this_p = [[(this_rp[0][i], this_rp[1][i]) for this_rp in rp] for rp in posterior_samples["params"]]
    samples = np.array(flow_plotter["sampler"](this_p, random.PRNGKey(i), 10000)[0])
    m1m2_samples.append(samples)
post_pd = np.array(m1m2_samples)

m1m2_samples = [ ]
import tqdm
flow_plotter = make_normalizing_flow(transform, theta_true, masks, mask_skips, permutations, bounds = bounds, context = test_lambda)
for i in tqdm.tqdm(range(ns)):
    this_p = [[(this_rp[0][i], this_rp[1][i]) for this_rp in rp] for rp in prior_samples["params"]]
    samples = np.array(flow_plotter["sampler"](this_p, random.PRNGKey(i), 10000)[0])
    m1m2_samples.append(samples)
prior_pd = np.array(m1m2_samples)
#thetas = np.array(thetas)

import matplotlib.pyplot as plt
plt.clf()

bins = np.linspace(min(theta_true[:,0]), max(theta_true[:,0]), 30+1)
for sample in tqdm.tqdm(prior_pd):
    _=plt.hist(sample[:,0], histtype = 'step', bins=bins, alpha = 0.1, color = 'green', density = True)
for sample in tqdm.tqdm(post_pd):
    _=plt.hist(sample[:,0], histtype = 'step', bins=bins, alpha = 0.1, color = 'b', density = True)
_=plt.hist(theta_true[:,0], histtype = 'step', bins=bins, linewidth = 2.0, color = 'orange', density = True)
plt.ylim(0,1.2)
plt.savefig("hmc_mc_more_complex.png") 
plt.clf()
bins = np.linspace(min(theta_true[:,1]), max(theta_true[:,1]), 30+1)
for sample in tqdm.tqdm(prior_pd):
    _=plt.hist(sample[:,1], histtype = 'step', bins=bins, alpha = 0.1, color = 'green', density = True)
for sample in tqdm.tqdm(post_pd):
    _=plt.hist(sample[:,1], histtype = 'step', bins=bins, alpha = 0.1, color = 'b', density = True)
_=plt.hist(theta_true[:,1], histtype = 'step', bins=bins, linewidth = 2.0, color = 'orange', density = True)
plt.ylim(0,10.2)
plt.savefig("hmc_chi_more_complex.png") 

