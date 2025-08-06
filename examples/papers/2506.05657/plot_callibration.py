import numpy as np
import sys
import glob

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

label = sys.argv[1]
files = gllob.glob(f"__run__/callibration_{label}_*.txt")

cs = np.linspace(0.1, 0.95, 10)
fig,ax = plt.subplots(1, figsize=(9*1.3,6*1.3), dpi =500)
for this_file in files:
    ecs = np.loadtxt(this_file)
    ec = ecs[:,int(sys.argv[2])]
    sigma = this_file[this_file.index('0_')+2:this_file.index('.txt')]
    ax.plot(cs, ec, label = r'$\sigma_{0}=$'+sigma)
ax.plot(cs,cs, color = 'black')
ax.set_xlabel("Nominal Coverage", fontsize=32)
ax.set_ylabel("Emperical Coverage", fontsize=32)
ax.legend(fontsize=25, loc = "upper left")
fig.savefig(f"__run__/callibration_{label}.png")
     


