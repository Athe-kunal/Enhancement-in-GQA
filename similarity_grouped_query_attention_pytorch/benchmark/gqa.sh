#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=saischin@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-9:59:00
#SBATCH --mem=16gb
#SBATCH --partition=gpu
#SBATCH --gpus v100:1
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=GQA
#SBATCH --output=gqa_out.txt
#SBATCH --error=gqa_err.txt
#SBATCH -A students

######  Module commands #####
module load python/gpu


######  Job commands go below this line #####
python ./gqa.py