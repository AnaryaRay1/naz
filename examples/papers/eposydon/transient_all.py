# POSYDON
from posydon.grids.psygrid import PSyGrid
from posydon.popsyn.synthetic_population import Population
from posydon.popsyn.synthetic_population import TransientPopulation
from posydon.popsyn.synthetic_population import Rates
from posydon.popsyn.binarypopulation import BinaryPopulation
from posydon.popsyn.transient_select_funcs import chi_eff, mass_ratio, m_chirp, BBH_selection_function
from posydon.popsyn.io import binarypop_kwargs_from_ini, simprop_kwargs_from_ini
from posydon.utils.common_functions import convert_metallicity_to_string


# Math and ML
import numpy as np
import pandas as pd
import os
import itertools
import h5py
import sys

folder = sys.argv[1]
output_path = os.path.join(folder,'all_BBH.h5')

files = ['1e-01_Zsun_population.h5',
         '1e-02_Zsun_population.h5',
         '1e-03_Zsun_population.h5',
         '1e-04_Zsun_population.h5',
         '1e+00_Zsun_population.h5',
         '2e-01_Zsun_population.h5',
         '2e+00_Zsun_population.h5',
         '4.5e-01_Zsun_population.h5']

for file in files:
    input_path = os.path.join(folder,file)
    pop = Population(input_path)
    tmp_data = pop.history.select(columns=['S1_state', 'S2_state', 'event'])
    S1_state = tmp_data['S1_state'] == 'BH'
    S2_state = tmp_data['S2_state'] == 'BH'
    state = tmp_data['event'] == 'CO_contact'
    indices = tmp_data.index
    del tmp_data
    mask = S1_state & S2_state & state
    selected_indices = indices[mask].to_list()
    print(f'File: {file}, Number of systems: {len(selected_indices)}')
    pop.export_selection(selected_indices,output_path, append=True)

BBH_pop = Population(filename=output_path, chunksize=10000)
BBH_pop.calculate_underlying_mass(f_bin=0.7, overwrite=True)
BBH_pop.calculate_formation_channels(mt_history=True)
BBH_mergers = BBH_pop.create_transient_population(BBH_selection_function, 'BBH')

BBH_mergers = TransientPopulation(filename=output_path, transient_name='BBH')

MODEL = {
    'delta_t' : 100, # Myr
    'SFR' : 'IllustrisTNG', # Neijssel2019, Madau+Fragos2017
    'sigma_SFR' : None,
}

# this convolves with complete multi-metallicity SFH - necessary for redshifts
rates = BBH_mergers.calculate_cosmic_weights('IllustrisTNG', MODEL_in=MODEL) 




