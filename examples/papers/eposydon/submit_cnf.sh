#!/bin/bash
#SBATCH -A b1094               # Allocation
#SBATCH -p ciera-gpu     # Queue
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00             # Walltime/duration of the job
#SBATCH --mem=40G          
#SBATCH --job-name=cnf_allz_265
#SBATCH --error=./logs/cnf_allz_265.err
#SBATCH --output=./logs/cnf_allz_265.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizabethteng@u.northwestern.edu

source ~/miniconda/etc/profile.d/conda.sh
source activate naz

python train_cnf_mle_q.py --hiddendims 256 256 128 128 128 128 --nflow 1 --batchsize 50000 \
--popsynth-file=./pops/allz_265_100k_q.h5 \
--dir='CNF_allz_265' --epistemic-only=True --suffix='_linearz'
