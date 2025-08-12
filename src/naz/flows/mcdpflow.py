"""Conditional and Unconditional Distribution Estimators using Normalizing Flows with Approximate (Monte-Carlo dropout) Uncertainty Quantification"""
__author__ = "Anarya Ray <anarya.ray@northwestern.edu>"

import torch
from torch import nn
from .flow import NormalizingFlow, set_device
from .transforms import inverse_bounding_transform
import copy
import numpy as np
import tqdm

def sample_uncached(transformed_pdf, *args, **kwargs):
    z_samples = transformed_pdf.base_dist.sample(args[0])
    for i, transform in enumerate(transformed_pdf.transforms):
        if hasattr(transform, "parts"):
            for j,t in enumerate(transform.parts):
                if i+j == 0:
                    y_samples = nn.Identity()(z_samples)
                y_samples = t._call(y_samples)
        else:
            if i==0:
                y_samples = nn.Identity()(z_samples)
            y_samples = transform._call(z_samples)
                
    return y_samples
            


class MCDPNormalizingFlow(NormalizingFlow):
    """
    Normalizing flows with Monte Carlo dropout inbetween flow layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_maker = args[0]
        assert kwargs["dropout_p"] not in [None, 0.0]

    def sample_uncertain(self, niter, *args, condition = None, **kwargs):
        self.train()
        if self.conditional:
           assert condition is not None
           pdf = self.flow_dist.condition(self.embedding_net(condition))
        else:
           pdf = self.flow_dist
        
        samples = [ ]
        for _ in tqdm.tqdm(range(niter)):
            if self.flow_maker != "cnf":
                y_samples = sample_uncached(pdf, *args, **kwargs)
            else:
                y_samples = pdf.sample(*args, **kwargs)
            x_samples = y_samples if self.bounds is None else inverse_bounding_transform(y_samples, self.bounds['low'], self.bounds["high"])
            samples.append(x_samples.cpu().detach().numpy())

        return np.array(samples)


