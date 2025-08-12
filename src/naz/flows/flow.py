"""Conditional and Unconditional Distribution Estimators using Normalizing Flows"""
__author__ = "Anarya Ray <anarya.ray@northwestern.edu>"

import torch
from torch import nn

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import PyroModule

try:
    from utils import set_device, device
except ImportError:
    from ..utils import set_device, device

from .transforms import bounding_transform, inverse_bounding_transform, masked_affine_autoregressive, neural_spline_autoregressive, neural_spline_coupling
from .continuous_transforms import continuous_free_form


flow_makers = {"maf": masked_affine_autoregressive, "nsa": neural_spline_autoregressive, "nsc": neural_spline_coupling, "cnf": continuous_free_form}


class NormalizingFlow(PyroModule):

    def __init__(self, flow_type, bounds, *flow_maker_args, embedding_net = None, **flow_maker_kwargs):
        super().__init__()
        assert flow_type in list(flow_makers.keys())
        flow_maker = flow_makers[flow_type]
        self.conditional = True if flow_maker_args[1] > 0 else False
        if embedding_net is not None:
            assert self.conditional
            self.embedding_net = embedding_net
        else:
            self.embedding_net = nn.Identity()
        self.bounds = bounds
        self.base_dist = dist.Normal(set_device(torch.zeros(flow_maker_args[0])), set_device(torch.ones(flow_maker_args[0])))
        self.flow, self.transforms, self.nets = flow_maker(*flow_maker_args, **flow_maker_kwargs)
        if self.conditional:
             self.flow_dist = dist.ConditionalTransformedDistribution(self.base_dist, self.flow) 
        else:
             self.flow_dist = dist.TransformedDistribution(self.base_dist, self.flow)


    def log_prob(self, x, *args, condition = None, **kwargs):
        '''
        Calculate log p(theta|lambda)

        ----------
        Parameters
        ----------

        x             :: torch.tensor (batch_dim, theta_dim)
                         batched theta values

        condition     :: torch.tensor (batch_dim, condition_dim)
                         batched lambda values corresponding to theta values


        -------
        Returns
        -------

        logprob       :: torch.tensor (batch_dim,)
                         log probability



        '''
        if self.bounds is None:
            y, log_jac = x, set_device(0.)
        else:
            y, log_jac = bounding_transform(x, self.bounds['low'], self.bounds['high'])
        if self.conditional:
            assert condition is not None
            pdf = self.flow_dist.condition(self.embedding_net(condition))
        else:
            pdf = self.flow_dist
        return pdf.log_prob(y, *args, **kwargs) + log_jac

    def bounded_log_prob(self, x, *args, condition = None, **kwargs):
        if self.bounds is None:
            return self.log_prob(x, *args, condition = condition, **kwargs)
        lp = torch.log(torch.zeros_like(x[:,0]))
        valid_args = torch.prod((x>self.bounds['low'].expand(x.shape))*(x<self.bounds['high'].expand(x.shape)), dim = 1).to(torch.bool)
        lp[valid_args] = self.log_prob(x[valid_args, :], *args, condition = condition, **kwargs)
        return lp

    
    def average_log_prob(self, x, *args, condition = None, **kwargs):
        return torch.mean(self.bounded_log_prob(x, *args, condition = condition, **kwargs))


    def sample(self, *args, condition = None, **kwargs):
        '''
        Sample theta~p(theta|lambda)

        ----------
        Parameters
        ----------


        condition     :: torch.tensor (condition_dim,)
                         lambda value corresponding to which theta samples are to be drawn

        Nsamples      :: list
                         shape of theta samples




        -------
        Returns
        -------

        samples       :: torch.tensor (Nsamples,theta_dim)
                         theta samples



        '''
        
        if self.conditional:
           assert condition is not None
           pdf = self.flow_dist.condition(self.embedding_net(condition))
        else:
           pdf = self.flow_dist
        y = pdf.sample(*args, **kwargs)
        return (y if self.bounds is None else inverse_bounding_transform(y, self.bounds['low'], self.bounds['high']))
    
    
    
     
     
