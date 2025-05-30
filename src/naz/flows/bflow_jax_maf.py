"Conditional and Un-conditional distribution estimators using Jax-based Masked Autoregressive Flows with Bayesian Uncertainty Quantification"
__author__ = "Anarya Ray <anarya.ray@northwestern.edu>"

import jax
import jax.numpy as jnp
import optax
from jax import random, value_and_grad
from jax import lax
from jax.flatten_util import ravel_pytree

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO

import h5py
import sys
import os
import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from functools import partial, reduce
import pickle
import copy

def torch_to_jax(torch_maf):
    masks, mask_skips, permutations, params, param_shapes = [ ], [ ], [ ], [ ], [ ]
    
    
    for flow_layer in torch_maf.flow_dist.transforms:
        this_params = [ ]
        this_param_shape = []
        
        arn = flow_layer.nn
        for layer in arn.layers:
            this_params.append((jnp.array(layer.weight.cpu().detach().numpy()), jnp.array(layer.bias.cpu().detach().numpy())))
            this_param_shape.append((jnp.array(layer.weight.cpu().detach().numpy()).shape, jnp.array(layer.bias.cpu().detach().numpy()).shape))
        param_shapes.append(this_param_shape)
        params.append(this_params)
        masks.append([jnp.array(this_mask.cpu().detach().numpy()) for this_mask in arn.masks])
        mask_skips.append(jnp.array(arn.mask_skip.cpu().detach().numpy()))
        permutations.append(jnp.array(arn.permutation.cpu().detach().numpy()))
    
        
            
    return params, param_shapes, masks, mask_skips, permutations

def sample_mask_indices(input_dim: int, hidden_dim: int, simple: bool = True) -> jnp.ndarray:
    indices = jnp.linspace(1, input_dim, hidden_dim)
    return jnp.round(indices) if simple else jnp.floor(indices) + jnp.array(np.random.bernoulli( indices - jnp.floor(indices)))

def create_mask(input_dim: int,
                    context_dim: int,
                    hidden_dims: List[int],
                    permutation: jnp.ndarray,
                    output_dim_multiplier: int) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
    var_index = jnp.empty_like(permutation, dtype=jnp.float32).at[permutation].set(jnp.arange(1, input_dim + 1))
    input_indices = jnp.concatenate([jnp.zeros(context_dim), var_index])
    
    if context_dim > 0:
        hidden_indices = [sample_mask_indices(input_dim, h) - 1 for h in hidden_dims]
    else:
        hidden_indices = [sample_mask_indices(input_dim - 1, h) for h in hidden_dims]
    
    output_indices = jnp.tile(var_index, output_dim_multiplier)
    mask_skip = (output_indices[:, None] > input_indices[None, :]).astype(jnp.float32)
    
    masks = [(hidden_indices[0][:, None] >= input_indices[None, :]).astype(jnp.float32)]
    for i in range(1, len(hidden_dims)):
        masks.append((hidden_indices[i][:, None] >= hidden_indices[i - 1][None, :]).astype(jnp.float32))
    masks.append((output_indices[:, None] > hidden_indices[-1][None, :]).astype(jnp.float32))
    return masks, mask_skip

@jax.jit
def masked_linear(params: Tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    W, b = params
    return jnp.dot(x, (W * mask).T) + b

@jax.jit    
def sigmoid_transform(x):
    return jax.nn.sigmoid(x)
    
@jax.jit
def sigmoid_inv(y):
    return -jnp.log((1 / y) - 1)
    
@jax.jit
def sigmoid_log_abs_det_jacobian(x):
    return (-jax.nn.softplus(-x) - jax.nn.softplus(x)).sum(-1)

@jax.jit
def tanh(args):
    return jax.nn.tanh(args)
    
@jax.jit
def bounding_transform(x, low, high):
    y = (x - jnp.expand_dims(low, 0))/(jnp.expand_dims((high-low),0))
    log_jac = - jnp.sum( jnp.log(y)+jnp.log1p(-y), axis=-1) - jnp.sum(jnp.log(high-low))
    return sigmoid_inv(y), log_jac
    
@jax.jit
def inverse_bounding_transform(y, low, high):
    x = sigmoid_transform(y)
    log_jac = sigmoid_log_abs_det_jacobian(y) + jnp.sum(jnp.log(high-low))
    return x*(jnp.expand_dims((high-low), 0)) + jnp.expand_dims(low, 0), log_jac

def make_conditional_autoregressive_nn(input_dim: int,
                                       context_dim: int,
                                       hidden_dims: List[int],
                                       param_dims: List[int] = [1,1],
                                       permutation: jnp.ndarray = None,
                                       skip_connections: bool = False,
                                       activation_fn: Callable = tanh,
                                       simple_masking: bool = True):
    output_multiplier = sum(param_dims)
    count_params = len(param_dims)
    all_ones = all(p == 1 for p in param_dims)
    def generate_mask(permutation = permutation):
        if permutation is None:
            permutation = jnp.array(np.random.permutation(input_dim))#jax.random.permutation(jax.random.PRNGKey(0), input_dim)
        else:
            permutation = jnp.array(permutation)

        masks, mask_skip = create_mask(input_dim, context_dim, hidden_dims, permutation, output_multiplier)
        return masks, mask_skip, permutation

    starts = jnp.cumsum(jnp.array([0] + param_dims[:-1]))
    ends = jnp.cumsum(jnp.array(param_dims))
    param_slices = list(zip(starts, ends))
    param_shapes = [((hidden_dims[0], input_dim + context_dim), (hidden_dims[0]))]
    for i in range(1, len(hidden_dims)):
        param_shapes.append(((hidden_dims[i], hidden_dims[i-1]), (hidden_dims[i])))
    param_shapes.append(((input_dim * output_multiplier, hidden_dims[-1]), (input_dim * output_multiplier)))

    def nn_fn(x: jnp.ndarray,
              params: List[Tuple[jnp.ndarray, jnp.ndarray]],
              masks: List[jnp.ndarray],
              mask_skip: List[jnp.ndarray],
              context: Optional[jnp.ndarray] = None,) -> Union[jnp.ndarray, Tuple[jnp.ndarray]]:
        if context is not None:
            context = jnp.broadcast_to(context, x.shape[:-1] + (context.shape[-1],))
            x_full = jnp.concatenate([context, x], axis=-1)
        else:
            x_full = x

        h = x_full
        for layer_params, mask in zip(params[:-1], masks[:-1]):
            h = activation_fn(masked_linear(layer_params, h, mask))

        out = masked_linear(params[-1], h, masks[-1])

        if skip_connections:
            skip_out = jnp.dot(x_full, params[-1][0] * mask_skip.T)  # no bias for skip
            out += skip_out

        if output_multiplier == 1:
            return out
        else:
            out = out.reshape(x.shape[:-1] + (output_multiplier, input_dim))
            if count_params == 1:
                return out
            elif all_ones:
                return jnp.split(out, out.shape[-2], axis=-2)
            else:
                return tuple(out[..., s:e, :] for (s, e) in param_slices)

    return jax.jit(nn_fn), param_shapes, generate_mask
    
def make_masked_affine_autoregressive_transform(nn_fn: Callable,
                          input_dim: int,
                          context: Optional[jnp.ndarray] = None):

    def forward_fn(xj: (jnp.ndarray, jnp.ndarray), args, context = context) -> Tuple[jnp.ndarray, jnp.array]:
        params, masks, mask_skip = args
        x, log_det_j = xj
        mean, log_scale = nn_fn(x, params, masks, mask_skip, context = context)
        log_scale = jnp.clip(log_scale, -5., 3.)
        y = jnp.squeeze(mean) + x * jnp.exp(jnp.squeeze(log_scale))
        return y, log_det_j + jnp.squeeze(log_scale).sum(-1)
    
    def inverse_fn(yj : (jnp.ndarray, jnp.ndarray), args, context = context) -> jnp.ndarray:
        params, masks, mask_skip, perm = args
        y, log_det_j = yj
        x = jnp.zeros_like(y)
        
        for idx in perm:
            mean, log_scale = nn_fn(x, params, masks, mask_skip, context = context)
            inverse_scale = jnp.squeeze(jnp.exp(-jnp.clip(log_scale[..., idx],-5.,3.)))#[..., 0]
            mean = jnp.squeeze(mean[..., idx])#[..., 0]
            x = x.at[...,idx].set((y[..., idx] - mean) * inverse_scale)
        
        log_scale = jnp.clip(log_scale, -5., 3.)
        return x, log_det_j + jnp.sum(jnp.squeeze(log_scale), axis = -1)
    return (jax.jit(forward_fn), jax.jit(inverse_fn))

def make_normalizing_flow(transform, x, masks, mask_skips, perms, bounds = None, context = None):
    if bounds is not None:
        x_inside = x[jnp.prod((x>jnp.expand_dims(bounds["low"], 0)) * (x<jnp.expand_dims(bounds["high"], 0)), axis=-1),:]
        x_inside, log_j_inv = bounding_transform(x_inside, bounds["low"], bounds["high"])
        print(jnp.isnan(x).any(), jnp.isnan(log_j_inv).any(),jnp.isinf(x_inside).any(), jnp.isinf(log_j_inv).any())
    else:
        x_inside = x
        log_j_inv = 0.
    if context is not None:
        inv_transform = jax.jit(partial(transform[1], context = context))
    else:
        inv_transform = transform[1]
    fwd_transform = transform[0]
        
    def log_prob(params):
        z, log_det_j = reduce(inv_transform, zip(reversed(params), reversed(masks), reversed(mask_skips), reversed(perms)), (x_inside, log_j_inv))
        return -jnp.sum(0.5*z**2, axis = -1) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi) - log_det_j
        
    def sample(params, rng_key, size):
        z = random.normal(rng_key, shape = (size, x.shape[-1]))
        if context is not None:
            assert len(context.shape) == 1
            this_fwd_transform = jax.jit(partial(fwd_transform, context = (jnp.ones(size)[:, None]*context[None,:])))
        y, log_j = reduce(this_fwd_transform, zip(params, masks, mask_skips), (z, -jnp.sum(0.5*z**2, axis = -1) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi) ))
        if bounds is not None:
            y, dlog_j =  inverse_bounding_transform(y, bounds["low"], bounds["high"])
            log_j += dlog_j
        return (y, log_j)

    return {"lp": jax.jit(log_prob), "sampler": jax.jit(sample, static_argnums = (2,))}

def bayesian_normalizing_flow(flow_lp, best_params, scale_max = 1.0, multi_scale = False, avg = False, fixed_scale = True):
    flat_params, unravel_fn = ravel_pytree(best_params)
    unravel_fn_jit = jax.jit(unravel_fn)
    
    print(f"model complexity: {jnp.size(flat_params)*(1 if not multi_scale else 2)}")
    
    @jax.jit
    def log_prob(params):
        return flow_lp(unravel_fn_jit(params)).sum() if not avg else flow_lp(unravel_fn_jit(params)).mean()
        
    def model(scale_max = scale_max, prior = False, anealed = False):
        scale = numpyro.deterministic("scale", scale_max) if fixed_scale else numpyro.sample("scale", dist.Uniform((jnp.zeros_like(flat_params) if multi_scale else 0), scale_max*jnp.ones_like(flat_params) if multi_scale else scale_max))
        standard_params = numpyro.sample("standart_params", dist.Uniform(-jnp.ones_like(flat_params), 1))
        random_params = numpyro.deterministic("params", flat_params * (1.0 + scale * standard_params))#a + (b-a) * standard_random_params)
        if not prior:
            log_l = numpyro.factor("log_l", log_prob(random_params))
           
        else:
            log_l = numpyro.factor("log_l", 0.0)
        return log_l
        
    def guide(scale_max = scale_max, scale_sigma_init = 0.1*scale_max, scale_mean_init = scale_max*0.5, svi_params = None):
        if svi_params is None:
            scale_sigma = numpyro.param("sigma_scale_q", jnp.ones_like(flat_params)*scale_sigma_init if multi_scale else scale_sigma_init, constraint = dist.constraints.interval(0.,scale_max))
            scale_mean = numpyro.param("mu_scale_q", jnp.ones_like(flat_params)*scale_mean_init if multi_scale else scale_mean_init, constraint = dist.constraints.interval(0.,scale_max))
            param_mean = numpyro.param("mu_param_q", flat_params, constraint = dist.constraints.interval(flat_params-jnp.absolute(flat_params)*scale_max, flat_params+jnp.absolute(flat_params)*scale_max))
        else:
            scale_sigma = numpyro.deterministic("sigma_scale_q", svi_params["sigma_scale_q"])
            scale_mean = numpyro.deterministic("mu_scale_q", svi_params["mu_scale_q"])
            param_mean = numpyro.deterministic("mu_param_q", svi_params["mu_param_q"])
        
        scale = numpyro.sample("scale", dist.TruncatedNormal(scale_mean, scale_sigma, low = 0., high = scale_max))
        param_sigma = numpyro.deterministic("sigma", jnp.absolute(param_mean)*scale)
        a = numpyro.deterministic("a", param_mean - param_sigma)
        b = numpyro.deterministic("b", param_mean + param_sigma)

        params = numpyro.sample("params", dist.TruncatedNormal(param_mean, param_sigma, low = a, high = b))
        return params
        
    def guided_model(scale_max = scale_max, scale_sigma_init = 0.1*scale_max, scale_mean_init = scale_max*0.5, svi_params = None, guide_fn = None):
        assert svi_params is not None and guide_fn is not None
        random_params = guide_fn(scale_max = scale_max, scale_sigma_init = 0.1*scale_max, scale_mean_init = scale_max*0.5, svi_params = svi_params)
        log_l = numpyro.factor("log_l", log_prob(random_params))
        
    return model, guide, guided_model, jax.jit(unravel_fn)

def train_maf(flow_train_lp, flow_test_lp,  param_shapes, lr=1e-3, num_epochs = 1024, patience = 64, lr_decay = 0.75, min_lr = 1e-7, min_epochs = 1024, clip_val = 1.0):
    params= [[(jnp.array(np.random.normal(size = this_p_shape[0]))*1e-5, jnp.array(np.random.normal(size = this_p_shape[1]))*1e-10) for this_p_shape in p_shape] for p_shape in param_shapes]
    optimizer = optax.inject_hyperparams(optax.adam)(lr)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state):
        def compute_loss(p):
            loss = flow_train_lp(p)            
            return -loss.mean()
        loss, grads = value_and_grad(compute_loss)(params)
        flat_grads, unravel_fn = ravel_pytree(grads)
        grads = unravel_fn(jnp.clip(flat_grads, -clip_val, clip_val))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    min_val_loss = jnp.inf
    min_train_loss = jnp.inf
    n_no_improve = 0
    train_loss, validation_loss = [ ], [ ]
    for epoch in range(num_epochs):
        params, opt_state, loss = step(params, opt_state)
        train_loss.append(loss)
        val_loss = -flow_test_lp(params).mean()
        validation_loss.append(val_loss)
        if epoch > min_epochs:
            if min_val_loss>val_loss:
                min_val_loss = val_loss
                n_no_improve = 0
                best_params = copy.deepcopy(params)
            if min_train_loss> loss:
                min_train_loss = loss
                best_train_params = copy.deepcopy(params)
            else:
                n_no_improve +=1
            if n_no_improve > patience/2:
                if opt_state.hyperparams['learning_rate'] > min_lr:
                    
                    opt_state.hyperparams['learning_rate'] *= lr_decay
                    n_no_improve = 0
                else:
                    print("min lr reached, continuing")
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: train_loss = {loss:.4f}, val_loss = {val_loss:.4f}, lr:{ opt_state.hyperparams['learning_rate']}, min_lr:{min_lr}, best_val_loss:{min_val_loss:.4f}, lr_decay = {lr_decay:0.4f}, best_train_loss: {min_train_loss}")

    return best_params, best_train_params, train_loss, validation_loss


def train_bayesian_flow_hmc(model, unravel_fn, scale_max=1.0, num_warmup = 1000, num_samples = 1000, target_accept = 0.8, num_chains = 1, checkpoint_file = 'checkpoint.pkl'):
    numpyro.set_host_device_count(num_chains)
    kernel = NUTS(model, target_accept_prob = target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, scale_max = scale_max)
    checkpt_state = mcmc.last_state
    
    with open(checkpoint_file, 'wb') as pf:
        pickle.dump(checkpt_state, pf)
    
    samples = mcmc.get_samples()
    samples["params"] = jax.jit(jax.vmap(unravel_fn))(samples["params"])
    return samples

def train_bayesian_flow_prior(model, unravel_fn, scale_max=1.0, num_samples = 1000):
    kernel = Predictive(model, num_samples=num_samples)
    samples = kernel(random.PRNGKey(0), scale_max = scale_max, prior = True)
    samples["params"] = jax.jit(jax.vmap(unravel_fn))(samples["params"])
    return samples
    

def train_bayesian_flow_svi(model, guide, unravel_func, scale_max = 1.0, num_steps = 1000, num_samples = 1000, step_size = 0.0005):
    optimizer = numpyro.optim.Adam(step_size=step_size)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), sum_steps)#, data)
    params = svi_result.params
    # get posterior samples
    predictive = Predictive(guide, params=params, num_samples=num_samples)
    posterior_samples = predictive(random.PRNGKey(1), data=None)
    posterior_samples["params"] = [unravel_fn(param) for param in posterior_samples["params"]]
    return posterior_samples

def train_bayesian_flow(model, unravel_fn, scale_max=1.0, num_warmup = 1000, num_samples = 1000, target_accept = 0.8, num_chains = 1, checkpoint_file = None, posterior_file = None, nbatch = 10):
    assert checkpoint_file is not None
    numpyro.set_host_device_count(num_chains)
    rng_key = random.PRNGKey(0)
    kernel = NUTS(model, target_accept_prob = target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=nbatch, num_chains=num_chains)
    if not os.path.isfile(checkpoint_file):
        mcmc.run(rng_key, scale_max = scale_max)
        posterior = mcmc.get_samples()
        checkpt_state = mcmc.last_state
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpt_state, f)

        with open(posterior_file, 'wb') as f:
            pickle.dump(posterior, f)
    else:
        with open(checkpoint_file, 'rb') as f:
            checkpt_state = pickle.load(f)

    with open(posterior_file, 'rb') as f:
        posterior = pickle.load(f)
    N_samples = len(posterior["scale"])
    while int(N_samples/num_chains)<num_samples:
        mcmc.post_warmup_state = checkpt_state
        mcmc.run(rng_key, scale_max = scale_max)
        new_posterior = mcmc.get_samples()
        for key in posterior:
            posterior[key] = jnp.concatenate((posterior[key], new_posterior[key]), axis = 0)
        checkpt_state = mcmc.last_state
        N_samples = len(posterior["scale"])
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpt_state, f)

        with open(posterior_file, 'wb') as f:
            pickle.dump(posterior, f)
        
    
    
  
    samples = posterior
    samples["params"] = jax.jit(jax.vmap(unravel_fn))(samples["params"])
    return samples
    
