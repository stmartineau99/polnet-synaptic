#!/bin/bash
 
#SBATCH -p standard96s:shared
#SBATCH -job-name=sbatch_test
#SBATCH -o %j.out
#SBATCH -e %j.out
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
 
source ~/.bashrc
micromamba activate polnet

python all_features_argument.py --out_dir test --ntomos 1 