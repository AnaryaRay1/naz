import numpy as np
import torch
import pickle
import corner
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import tqdm
import sys

import torch
from naz.utils import set_device
from naz.flows.flow import NormalizingFlow
from naz.trainers.train_flows import train
import h5py


import pandas as pd

index = int(sys.argv[1])
np.random.seed(69+index)
fthin = int(sys.argv[2])

if not os.path.exists(f"aleatoric_{fthin}/"):
    os.mkdir(f"aleatoric_{fthin}")

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
  theta[:,0 ] = np.log((m1*m2)**(3/5)/((m1+m2)**(1/5)))
  theta[:, 1] = theta_true[:,-2]

  
  data_true = theta.copy()
  theta = [ ]
  
  test_lambda = hf["test_lambda"][()]


hidden_dims = [150, 150, 150]

num_layers = 16

label = f"{int(hidden_dims[0])}_{len(hidden_dims)}_{int(num_layers)}_{index}"

flow = NormalizingFlow('maf', None, 2,2, hidden_dims, num_layers)#, activation = nn.ReLU)


model, history, history_val, best_mse,best_epoch = train(flow, set_device(thetas), set_device(lambdas), train_frac = 0.89, patience = 64, lr = 1e-3, min_lr = 1e-9, num_epochs = 2*4096, batch_frac = 0.1, lr_decay = 0.5)

with open(f'aleatoric_{fthin}/inference_mle_01_1_prod_2d_{label}_mcchi.pkl','wb') as f:
    pickle.dump(model,f)

with open(f'aleatoric_{fthin}/inference_mle_01_1_prod_2d_{label}_mcchi.pkl','rb') as f:
    model = pickle.load(f)

bounds = [(min(data_true[:,i]),max(data_true[:,i])) for i in range(data_true.shape[-1])]
num_bins = 30

# Predict new simulation and compare MLE with truth
data_pred_mle = model.sample([len(data_true)],condition = set_device([test_lambda]) ).cpu().detach().numpy()# condition = set_device([test_chib, test_alpha]))
fig = corner.corner(data_true, hist_kwargs = {'density': True}, color = 'orange',range = bounds, bins = num_bins+1 )
fig = corner.corner(data_pred_mle, hist_kwargs = {'density': True}, color= 'b', fig=fig, labels = [r'$\log\frac{\mathcal{M}}{M_{\odot}}$', r'$\chi_{eff}$'], range = bounds, bins = num_bins+1 )
fig.savefig(f'aleatoric_{fthin}/compare_mle_CE_01_1_prod_2d_{label}_mcchi.png')
