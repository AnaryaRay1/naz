import pandas as pd
import h5py
import numpy as np


np.random.seed(69)
model_file = sys.argv[1]
lambdas = [ ]
thetas = [ ]
chi_bs=np.array([0.0,0.1,0.2,0.5])
alphas = np.array([0.2,0.5,1.0,2.0,5.0])
n_cbc_per_pop = 10000
test_chib = 0.1
test_alpha = 2.0
chi_bs, alphas = np.meshgrid(chi_bs,alphas)
chi_bs, alphas = chi_bs.flatten(),alphas.flatten()

for i,(this_chib, this_alpha) in enumerate(zip(tqdm.tqdm(chi_bs),alphas)):
    if this_alpha<1:
        alpha_key = f'0{int(this_alpha*10)}'
    else:
        alpha_key = f'{int(this_alpha)}0'
    df_CE = pd.read_hdf(model_file,key=f'CE/chi0{int(this_chib*10.)}/alpha'+alpha_key)
    events_CE = np.concatenate(tuple([df_CE[param].to_numpy()[:,None] for param in ["m1", "m2", "chieff", "z"]]), axis = -1)
    this_thetas = events_CE[np.random.choice(np.arange(len(events_CE)), p=df_CE['weight'].to_numpy()/sum(df_CE['weight'].to_numpy()), size = n_cbc_per_pop, replace = False), :]
    this_thetas_dense = events_CE[np.random.choice(np.arange(len(events_CE)), p=df_CE['weight'].to_numpy()/sum(df_CE['weight'].to_numpy()), size = int(1e6), replace = True), :]
    this_lambdas = np.array([[this_chib, this_alpha]])*(np.ones(n_cbc_per_pop)[:,None])
    if this_alpha == test_alpha and test_chib == this_chib:
        data_true = this_thetas.copy()
        np.savetxt("__run__/test_set_dense.txt", this_thetas_dense)
        continue
    if i== 0:
        thetas = this_thetas.copy()
        lambdas = this_lambdas.copy()
    else:
        thetas = np.concatenate((thetas, this_thetas), axis = 0)
        lambdas = np.concatenate((lambdas, this_lambdas), axis = 0)

with h5py.File("__run__/CE_Bavera_2020.h5", "w") as hf:
    hf.create_dataset("train_theta", data = thetas)
    hf.create_dataset("train_lambda", data = lambdas)
    hf.create_dataset("test_theta", data = data_true)
    hf.create_dataset("test_lambda", data = np.array([test_chib, test_alpha]))
