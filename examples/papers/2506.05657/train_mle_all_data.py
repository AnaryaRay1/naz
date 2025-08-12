import numpy as np
import torch
import pickle
import corner
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import tqdm
import sys
#source = os.getcwd()+'/src/sbi/'
#sys.path.append(source)

import torch
from naz.utils import set_device
from naz.flows.flow import NormalizingFlow
from naz.trainers.train_flows import train

import pandas as pd

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

with h5py.File("__run__/CE_Bavera_2020.h5", "r") as hf:
  np.random.seed(69)
  theta_train = hf["train_theta"][()]
  thetas = np.zeros((len(theta_train),2))
  m1, m2 = theta_train[:,0], theta_train[:, 1]
  thetas[:, 0] = np.log((m1*m2)**(3/5)/((m1+m2)**(1/5)))
  thetas[:, 1] = theta_train[:,-2]
  theta_train = [ ]
  lambdas = hf["train_lambda"][()]
  theta_true = hf["test_theta"][()]

  theta = np.zeros((len(theta_true),2))
  m1, m2 = theta_true[:,0], theta_true[:, 1]
  theta[:,0 ] = np.log( (m1*m2)**(3/5)/((m1+m2)**(1/5)))
  theta[:, 1] = theta_true[:,-2]


  theta_true = theta.copy()
  theta = [ ]

  test_lambda = hf["test_lambda"][()]


print(thetas.shape, lambdas.shape)

hidden_dims = [150, 150, 150]
# hidden_dims = [64, 64, 64]
num_layers = 16

label = f"{int(hidden_dims[0])}_{len(hidden_dims)}_{int(num_layers)}"
#label = f"{int(hidden_dims[0])}_{len(hidden_dims)}_{int(num_layers)}_cnf"


flow = NormalizingFlow('maf', None, 2,2, hidden_dims, num_layers)#, activation = nn.ReLU)
#flow = NormalizingFlow('cnf', None, 2,2, hidden_dims)


model, history, history_val, best_mse,best_epoch = train(flow, set_device(thetas), set_device(lambdas), train_frac = 0.89, patience = 64, lr = 1e-3, min_lr = 1e-9, num_epochs = 2*4096, batch_frac = 0.05, lr_decay = 0.5)

model_dir = '__models__'
with open(f'inference_mle_01_1_prod_2d_{label}_mcchi.pkl','wb') as f:
    pickle.dump(model,f)

model_dir = '__models__'
with open(f'inference_mle_01_1_prod_2d_{label}_mcchi.pkl','rb') as f:
    model = pickle.load(f)

plot_dir = '__plots__'
bounds = [(min(theta_true[:,i]),max(data_true[:,i])) for i in range(data_true.shape[-1])]
num_bins = 30

# Predict new simulation and compare MLE with truth
data_pred_mle = model.sample([n_cbc_per_pop],condition = set_device(test_lambda) ).cpu().detach().numpy()# condition = set_device([test_chib, test_alpha]))
fig = corner.corner(theta_true, hist_kwargs = {'density': True}, color = colors[1],range = bounds, bins = num_bins+1 )
fig = corner.corner(data_pred_mle, hist_kwargs = {'density': True}, color= colors[0], fig=fig, labels = [r'$\log\frac{\mathcal{M}}{M_{\odot}}$', r'$\chi_{eff}$'], range = bounds, bins = num_bins+1 )
fig.savefig(f'compare_mle_CE_01_1_prod_2d_{label}_mcchi.png')
