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
from naz.utils import set_device, device
from naz.flows.flow import NormalizingFlow
from naz.trainers.train_flows import train, train_lightning

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

np.random.seed(69)
model_file = sys.argv[1]
lambdas = [ ]
thetas = [ ]
chi_bs=np.array([0.0,0.1,0.2,0.5])
alphas = np.array([0.2,0.5,1.0,2.0,5.0])
n_cbc_per_pop = 10000
test_chib = 0.1
test_alpha = 1.0
chi_bs, alphas = np.meshgrid(chi_bs,alphas)
chi_bs, alphas = chi_bs.flatten(),alphas.flatten()

ignore_alphas = [100,200]
ignore_chi_bs = [100.1,100.2]
for i,(this_chib, this_alpha) in enumerate(zip(tqdm.tqdm(chi_bs),alphas)):
    if this_alpha<1:
        alpha_key = f'0{int(this_alpha*10)}'
    else:
        alpha_key = f'{int(this_alpha)}0'
    df_CE = pd.read_hdf(model_file,key=f'CE/chi0{int(this_chib*10.)}/alpha'+alpha_key)
    events_CE = np.concatenate(tuple([df_CE[param].to_numpy()[:,None] for param in ["m1", "m2", "chieff", "z"]]), axis = -1)
    this_thetas = events_CE[np.random.choice(np.arange(len(events_CE)), p=df_CE['weight'].to_numpy()/sum(df_CE['weight'].to_numpy()), size = n_cbc_per_pop, replace = False), :]
    this_lambdas = np.array([[this_chib, this_alpha]])*(np.ones(n_cbc_per_pop)[:,None])
    if this_alpha == test_alpha and test_chib == this_chib:
        data_true = this_thetas.copy()
        continue
    elif this_alpha in ignore_alphas and this_chib in ignore_chi_bs:
        print("skipped")
        continue
    else:
        pass
    if i==0:
        thetas = this_thetas.copy()
        lambdas = this_lambdas.copy()
    else:
        thetas = np.concatenate((thetas, this_thetas), axis = 0)
        lambdas = np.concatenate((lambdas, this_lambdas), axis = 0)

thetas_new = thetas.copy()[:,:2]
m1, m2 = thetas_new[:,0], thetas_new[:, 1]
thetas_new[:,0] = np.log((m1*m2)**(3/5)/((m1+m2)**(1/5)))
#thetas_new[:,0] = np.log(m1)
thetas_new[:,1] = thetas[:,2]
thetas = thetas_new.copy()

thetas_new = data_true.copy()[:,:2]
m1, m2 = thetas_new[:,0], thetas_new[:, 1]
thetas_new[:,0] = np.log((m1*m2)**(3/5)/((m1+m2)**(1/5)))
#thetas_new[:,0] = np.log(m1)
thetas_new[:,1] = data_true[:,2]
data_true = thetas_new.copy()
thetas_new = [ ]


print(thetas.shape, lambdas.shape)


label = "cnf"
'''
flow = NormalizingFlow("cnf", None, 2, 2, [128, 64, 64], 1)

model = train_lightning(flow, set_device(thetas), set_device(lambdas), num_epochs = 1024)#opt = torch.optim.AdamW, lambda_l2 = 1e-5, train_frac = 0.89, patience = 64, lr = 2.0e-3, min_lr = 1e-8, num_epochs = 4096, batch_frac = 0.05)


model_dir = '__models__'
with open(f'inference_mle_01_1_prod_2d_{label}_mcchi.pkl','wb') as f:
    pickle.dump(model,f)
'''
model_dir = '__models__'
with open(f'inference_mle_01_1_prod_2d_{label}_mcchi.pkl','rb') as f:
    model = pickle.load(f)

model.to(device)
plot_dir = '__plots__'
bounds = [(min(data_true[:,i]),max(data_true[:,i])) for i in range(data_true.shape[-1])]
#bounds = {k:v.cpu().detach().numpy() for k,v in bounds.items()}
#bounds = [(bounds['low'][i], bounds['high'][i]) for i in range(thetas.shape[-1])]
print(bounds)
num_bins = 30

# Predict new simulation and compare MLE with truth
data_pred_mle = model.sample([n_cbc_per_pop],condition = set_device(np.array([test_chib, test_alpha])[None,:]*np.ones((n_cbc_per_pop,1))) ).cpu().detach().numpy()# condition = set_device([test_chib, test_alpha]))
print(np.where(np.isnan(data_pred_mle)))
print(data_pred_mle)
fig = corner.corner(data_true, hist_kwargs = {'density': True}, color = colors[1],range = bounds, bins = num_bins+1 )
fig = corner.corner(data_pred_mle, hist_kwargs = {'density': True}, color= colors[0], fig=fig, labels = [r'$\log\frac{\mathcal{M}}{M_{\odot}}$', r'$\chi_{eff}$'], range = bounds, bins = num_bins+1 )
fig.savefig(f'compare_mle_CE_01_1_prod_2d_{label}_mcchi.pdf')
