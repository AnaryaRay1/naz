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
filename = f"__run__/{label}_coverage.txt"
legend = (sys.argv[2]=="true")
nQs = [25, 49, 100, 400]
sm = sys.argv[3]
cs = np.linspace(0.1, 0.95, 10)
fig,ax = plt.subplots(1, figsize=(9*1.3,9*1.3), dpi =500)
ecs = np.loadtxt(filename)
for i, ec in enumerate(ecs.T):
    ax.plot(cs, ec, label = r'$n_{Q}=$'+str(int(nQs[i])), linewidth=1.3)
ax.plot(cs,cs, '--', color = 'black', linewidth = 2.0)
ax.set_xlabel("Nominal Coverage", fontsize=32)
ax.set_ylabel("Empirical Coverage", fontsize=32)
ax.set_title(r"$\sigma_{0}=$"+sm, fontsize = 40)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.grid()
if legend:
    ax.legend(fontsize=25, loc = "upper left")
fig.tight_layout()
fig.savefig(f"__run__/calibration_{label}.png")

     


