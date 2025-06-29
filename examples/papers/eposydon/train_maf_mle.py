import numpy as np
import pickle
import corner
import os
import tqdm
import sys
import h5py

import torch
from naz.utils import set_device
from naz.flows.flow import NormalizingFlow
from naz.trainers.train_flows import train


import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser(description = "Train MAF MLE")

parser.add_argument('--epistemic-only', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether or not to re-run on the same dataset')

parser.add_argument('--nhidden', type=int, default=512,
                    help='number of hidden units')

parser.add_argument('--nlayer', type=int, default=5,
                    help='number of hidden layers')

parser.add_argument('--nflow', type=int, default=16,
                    help='number of flow layers')

parser.add_argument('--index', type=int, default=0,
                    help='index of run')


parser.add_argument('--fthin', type=int, default=1,
                    help='number of batches to split dataset into')


parser.add_argument('--popsynth-file', type=str,
                    help='h5 file containing synthesized binaries')

args = parser.parse_args()

index = int(args.index)
avg = args.epistemic_only
fthin = int(args.fthin)
popsynth_file = args.popsynth_file
nh = int(args.nhidden)
nhl = int(args.nlayer)
num_layers = int(args.nflow)
print(index, avg, fthin)

outdir = f"mle_rerunrs_{'epistemic' if avg else 'aleatoric'}_{fthin}_4p/"

if not os.path.exists(outdir):
    try:
        os.mkdir(outdir)
    except:
        pass


with h5py.File(popsynth_file, "r") as hf:
  np.random.seed(69+(index if not avg else 0 ))
  theta_train = hf["theta"][()]
  N = len(theta_train)
  rand_indices = np.random.choice(N, size = int(N/fthin))
  theta_train = theta_train[rand_indices,:]
  thetas = theta_train.copy()
  thetas[:,:2] = np.log(thetas[:,:2])
  lambdas = hf["lambda"][()][rand_indices,:]


hidden_dims = [nh for i in range(nhl)]

num_layers = 16

label = f"{int(hidden_dims[0])}_{len(hidden_dims)}_{int(num_layers)}_{index}_4p"

flow = NormalizingFlow('maf', None, thetas.shape[-1],lambdas.shape[-1], hidden_dims, num_layers)#, activation = nn.ReLU)


model, history, history_val, best_mse,best_epoch = train(flow, set_device(thetas), set_device(lambdas), train_frac = 0.89, patience = 64, lr = 1e-3, min_lr = 1e-9, num_epochs = 4096, batch_frac = 0.05, lr_decay = 0.5, return_final = True)

with open(outdir+f'inference_mle_{label}.pkl','wb') as f:
    pickle.dump(model,f)
'''
with open(outdir+f'inference_mle_{label}.pkl','rb') as f:
    model = pickle.load(f)

bounds = [(min(data_true[:,i]),max(data_true[:,i])) for i in range(data_true.shape[-1])]
num_bins = 30

# Predict new simulation and compare MLE with truth
data_pred_mle = model.sample([len(data_true)],condition = set_device([test_lambda]) ).cpu().detach().numpy()# condition = set_device([test_chib, test_alpha]))
fig = corner.corner(data_true, hist_kwargs = {'density': True}, color = 'orange',range = bounds, bins = num_bins+1 )
fig = corner.corner(data_pred_mle, hist_kwargs = {'density': True}, color= 'b', fig=fig, labels = [r"$\log\frac{m_1}{M_{\odot}}$", r'$\log\frac{m_2}{M_{\odot}}$', r"$\chi_{eff}$", r"$z$"], range = bounds, bins = num_bins+1 )
fig.savefig(outdir+f'compare_mle_CE_01_1_prod_2d_{label}_mcchi.png')
'''
