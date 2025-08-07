from posydon.popsyn.synthetic_population import TransientPopulation
import numpy as np
import h5py
import copy
import corner
import sys
from astropy.cosmology import Planck15
from astropy import units as u

trans_pop = TransientPopulation(filename=sys.argv[1], transient_name='BBH')
pop = trans_pop.population

thetas = np.array([pop[key].to_numpy()[:,None].flatten() for key in ['S1_mass', 'S2_mass', 'chi_eff','time']]).T

n_cbc_per_pop = 1000000
indices = np.random.choice(len(pop), size = n_cbc_per_pop )

thetas = thetas[indices,:]

thetas_copy = copy.deepcopy(thetas)
thetas[:,2] = thetas_copy[:,-1]
thetas[:,-1] = thetas_copy[:, 2]
thetas[:,0] = np.maximum(thetas_copy[:,0], thetas_copy[:,1])
thetas[:,1] = np.minimum(thetas_copy[:,0], thetas_copy[:,1])
highs = np.quantile(thetas, 0.95, axis = 0)*1.3
lows = np.min(thetas, axis = 0)

limits = [(low, high) for low, high in zip(lows, highs)]
print(np.where(np.isnan(thetas)))

with h5py.File(sys.argv[2], "w") as hf:
    hf.create_dataset("parameters", data = thetas)
