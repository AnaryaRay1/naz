import numpy as np
import pickle
import corner
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import tqdm
import sys

from naz.utils import set_device
import torch
from naz.flows.flow import NormalizingFlow
from naz.trainers.train_flows import train
import h5py


import pandas as pd


hidden_dims = [150, 150, 150]

num_layers = 10

label = f"{int(hidden_dims[0])}_{len(hidden_dims)}_{int(num_layers)}"

data = np.genfromtxt('__run__/posterior_samples_narrow_spin_prior_170817.dat',names=True)
q = data['q']
lambda_t = data['lambdat']
var_q = np.std(q)
var_lt =np.std(lambda_t)
data = np.array([q/var_q, lambda_t/var_lt]).T
bounds = [[0,1.1/var_q], [0., max(lambda_t)*1.1/var_lt]]
bounds_dict = {'low':set_device([bounds[0][0], bounds[1][0]]), 'high': set_device([bounds[0][1], bounds[1][1]])}
flow = NormalizingFlow('maf', bounds_dict, 2,0, hidden_dims, num_layers)#, activation = nn.ReLU)
data_true = data.copy()
'''
model, history, history_val, best_mse,best_epoch = train(flow, set_device(data), set_device(np.ones_like(data)), train_frac = 0.89, patience = 64, lr = 1e-3, min_lr = 1e-9, num_epochs = 2*4096, batch_frac = 0.1, lr_decay = 0.5)

with open(f'__run__/inference_mle_unsupervised_{label}_bounded.pkl','wb') as f:
    pickle.dump(model,f)
'''
with open(f'__run__/inference_mle_unsupervised_{label}_bounded.pkl','rb') as f:
    model = pickle.load(f)

num_bins = 30

# Predict new simulation and compare MLE with truth
data_pred_mle = model.sample([len(data_true)],).cpu().detach().numpy()#condition = set_device([test_lambda]) ).cpu().detach().numpy()# condition = set_device([test_chib, test_alpha]))
fig = corner.corner(data_true, hist_kwargs = {'density': True}, color = 'orange',range = bounds, bins = num_bins+1 )
fig = corner.corner(data_pred_mle, hist_kwargs = {'density': True}, color= 'b', fig=fig, labels = [r'$q$', r'$\tilde{\Lambda}$'], range = bounds, bins = num_bins+1 )
fig.savefig(f'__run__/compare_mle_unsupervised_{label}_bounded.png')
