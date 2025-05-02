"Conditional and Unconditional Density Estimation with Bayesian Uncertainty Quantification"
__author__ = "Anarya Ray <anarya.ray@northwestern.edu>"

from .flow import NormalizingFlow, set_device, device
from .transforms import bounding_transform
import pyro.distributions as dist
from pyro.distributions.transforms import ComposeTransform, SigmoidTransform, AffineTransform
from priors.TruncatedNormal import TruncatedNormal
try:
    from priors.TruncatedNormal import TruncatedNormal
except ImportError:
    from ..priors.TruncatedNormal import TruncatedNormal

import pyro
from torch import nn
import torch
import warnings

class BayesianNormalizingFlow(NormalizingFlow):

    def __init__(self, mle_flow, *args, prior_dist = 'Uniform', scale_max = 0.1, set_grad = True, **kwargs):
        super(BayesianNormalizingFlow, self).__init__(*args, **kwargs)
        self.mle_flow = mle_flow
        self.prior_dist = prior_dist
        self.scale_max = set_device(scale_max)
        self.set_grad = set_grad
        self.n_params = None
        self.param_bounds = {}

    def draw_param(self, name, param, mean, sigma, guide = False, guide_params = None):
        a = mean - sigma                
        b = mean + sigma
        if guide:
            mean = pyro.param(f"{name}_mean_q", set_device(mean) if guide_params is None else guide_params[f"{name}_mean_q"], constraint = constraints.interval(a,b))
            this_dist = TruncatedNormal(set_device(mean), set_device(sigma),set_device(a.clone().detach()),set_device(b.clone.detach()))     
        elif self.prior_dist == 'StandardNormal':
            this_dist = dist.Normal(set_device(0.0).expand(mean.shape), set_device(1.0).expand(mean.shape))
        elif self.prior_dist == 'Normal':
            this_dist = dist.Normal(set_device(mean.clone().detach()), set_device(sigma.clone().detach()))
        elif self.prior_dist == 'Uniform':
            this_dist = dist.Uniform(set_device(a.clone().detach()),set_device(b.clone().detach()))
        elif self.prior_dist == 'TruncNorm':
            this_dist = TruncatedNormal(set_device(mean.clone().detach()), set_device(sigma.clone().detach()),set_device(a.clone().detach()),set_device(b.clone().detach()))
        else:
            raise
        sampled_param = pyro.sample(f"{name}", this_dist.to_event(param.dim()))
        return sampled_param, a, b

    def set_param(self, param, new_param, grad = None):
        with torch.no_grad():
            param = set_device(param)
            param.copy_(new_param)
        if grad is not None and self.set_grad:
            param.requires_grad_(True)
            param.grad.copy_(set_device(grad.deatch().clone()))

    def prior_model(self, guide = False, set_param_bounds = False, guide_params = None):
        n = 0
        if guide:
            scale_mean = pyro.param("scale_mu_q", self.scale_max/2 if guide_params is None else guide_params["scale_mu_q"], constraint = constraints.interval(0., self.scale_max))
            scale_sigma = pyro.param("scale_sigma_q", self.scale_max/4 if guide_params is None else guide_params["scale_sigma_q"], constraint = constraints.interval(0., self.scale_max))
            scale = pyro.sample("scale", TruncatedNormal(set_device(scale_mean), set_device(scale_sigma), set_device(0.), set_device(self.scale_max)))
        else:
            scale = pyro.sample("scale", dist.Uniform(set_device(0.), set_device(self.scale_max)))
        
        for i,(t1, t2) in enumerate(zip(self.flow_dist.transforms, self.mle_flow.flow_dist.transforms)):
            parameters_mle = dict(t2.named_parameters())
            for name, param in t1.named_parameters():
                mean = parameters_mle[name].data
                n+=mean.numel()
                sigma = scale.expand(param.shape)*abs(mean)
                sampled_param, a, b = self.draw_param(f"flow_{i}_{name}", param, mean, sigma, guide = guide, guide_params = guide_params)
                if set_param_bounds:
                    self.param_bounds[f"flow_{i}_{name}"] = (a,b)
                    continue

                if not guide:
                    self.set_param(param, sampled_param, grad = parameters_mle[name].grad)

        if self.embedding_net is None or isinstance(self.embedding_net, nn.Identity):
            pass
        else:
            parameters_mle = dict(self.mle_flow.embedding_net.named_parameters())
            for name, param in self.embedding_net.named_parameters():
                mean = parameters_mle[name].data
                n+=mean.numel()
                sigma = scale.expand(param.shape)*abs(mean)
                sampled_param, a, b = self.draw_param(f"embedding_{0}_{name}", param, mean, sigma, guide = guide)
                if set_param_bounds:
                    self.param_bounds[f"embedding_{0}_{name}"] = (a,b)
                    continue
                
                if not guide:
                    self.set_param(param, sampled_param, grad = parameters_mle[name].grad)

        if self.n_params is None:
            self.n_params = n
        pass
   
    def model(self, theta, condition = None):
        self.prior_model(guide = False, set_param_bounds = False)
        if self.bounds is None:
            theta, log_jac = theta, set_device(0.)
        else:
            theta, log_jac = bounding_transform(theta, self.bounds['low'], self.bounds['high'])
        with pyro.plate("data", theta.shape[0]):
            if not self.conditional:
                pdf = self.flow_dist
            else:
                assert condition is not None
                pdf = self.flow_dist.condition(self.embedding_net(condition))
            obs = pyro.sample("log_l", pdf, obs = theta)
            pyro.factor("log_jac_bounding", log_jac)

        return obs

    def svi_guide(self, guide_params = None):    
        self.prior_model(guide = True, set_param_bounds = False, guide_params = guide_params)

    def param_transforms(self):
        if self.param_bounds == {}:
            self.prior_model(guide = False, set_param_bounds = True)
        return {k : ComposeTransform([SigmoidTransform(), AffineTransform(loc=a, scale=(b - a)) ]) for k, (a,b) in self.param_bounds.items()}














