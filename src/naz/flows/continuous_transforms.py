try:
    from utils import device, set_device
except ImportError:
    from ..utils import device, set_device


import torch
from torch import nn
from pyro.distributions.transforms  import ComposeTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.conditional import ConditionalTransformModule, ConditionalComposeTransformModule
import pyro.distributions as dist
from torch.distributions import constraints

from torchdyn.core import NeuralODE
from torchdyn.models import CNF
from torchdyn.nn import DataControl, DepthCat, Augmenter
from torchdyn.datasets import *
from torchdyn.utils import *


from functools import partial
from contextlib import contextmanager
import os
import sys
import copy

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class ConditionalFCNN(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dims, act = nn.Softplus(), dropout_p = 0.0):
        super().__init__()
        layers = [nn.Linear(input_dim + context_dim, hidden_dims[0]), act]
        if dropout_p is not None:
            layers.append(nn.Dropout(p = dropout_p))
        for i, hidden_dim in enumerate(hidden_dims[1:]):
            layers.append(nn.Linear(hidden_dims[i], hidden_dim))
            layers.append(act)
            if dropout_p is not None:
                layers.append(nn.Dropout(p = dropout_p))
        layers.append(nn.Linear(hidden_dims[-1], input_dim))
        self.nn = nn.Sequential(*layers)

    def _forward(self, x):
        return self.nn(x)

    def _conditioned_forward(self, x, context = None):
        x = torch.cat([x, context], dim = -1)
        return self._forward(x)

    def forward(self, x):
        raise NotImplementedError

class FCNN(ConditionalFCNN):
    def __init__(self, input_dim, hidden_dims, act = nn.Softplus(), dropout_p = 0.0):
        super().__init__(self, input_dim, 0, hidden_dims, act = act, dropout_p = dropout_p)

    def forward(self, x):
        return self._forward(x)


class FFJORDTransform(TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    
    def __init__(self, net, input_dim, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4):
        super(FFJORDTransform, self).__init__()
        noise_dist = dist.MultivariateNormal(set_device(torch.zeros(input_dim)), set_device(torch.eye(input_dim)))
        cnf = CNF(net, trace_estimator=self.hutch_trace, noise_dist=noise_dist).to(device)
        with suppress_stdout():
            self.nde = NeuralODE(cnf, solver=solver, sensitivity=sensitivity, atol=atol, rtol=rtol).to(device)
        self.model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1),
                      self.nde).to(device)
        self._cached_logdet = None
    
    def hutch_trace(self, x_out, x_in, noise=None, **kwargs):
        """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
        jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
        trJ = torch.einsum('bi,bi->b', jvp, noise)
        return trJ

    def _inverse(self, x):  # forward: z1 → z0
        t_range, xtrJ = self.model(x)
        self.nde.nfe = 0
        self._cached_logdet = xtrJ[-1,:,0]
        return xtrJ[-1,:,1:]

    def _call(self, z):  # inverse: z0 → z1
        t_span = torch.linspace(1,0,2).to(device)
        t, xtrJ = self.model[1](self.model[0](z), t_span)
        self.nde.nfe = 0
        self._cached_logdet = xtrJ[-1,:,0]
        return xtrJ[-1,:,1:]


    def log_abs_det_jacobian(self, x, y):
        return self._cached_logdet


class ConditionalFFJORDTransform(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, net, input_dim, context_dim, **kwargs):
        super().__init__()
        self.unconditional_transform = FFJORDTransform(net, input_dim, **kwargs)


    def condition(self, context):
        self.unconditional_transform.model[1].vf.vf.vf.net.forward = partial(self.unconditional_transform.model[1].vf.vf.vf.net._conditioned_forward, context = context)
        return self.unconditional_transform


def continuous_free_form(input_dim, condition_dim, hidden_dims, num_blocks, activation = nn.Softplus(), use_batchnorm = False, dropout_p = None, **kwargs):
    nets, transforms = [], []
    for _ in range(num_blocks):
        if condition_dim == 0:
            net = FCNN(input_dim, hidden_dims, act = activation,dropout_p = dropout_p).to(device) 
            nets.append(net)
            transform = FFJORDTransform(net, input_dim, **kwargs).to(device)
            transforms.append(transform)
        else:
            net = ConditionalFCNN(input_dim, condition_dim, hidden_dims, act = activation, dropout_p = dropout_p).to(device)
            nets.append(net)
            transform = ConditionalFFJORDTransform(net, input_dim, condition_dim, **kwargs).to(device)
            transforms.append(transform)

    flow = ComposeTransformModule(transforms).to(device) if condition_dim <= 0 else ConditionalComposeTransformModule(transforms).to(device)
    return flow, transforms, nets
    
    




