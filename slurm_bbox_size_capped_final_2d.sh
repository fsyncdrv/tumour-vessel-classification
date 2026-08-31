#!/bin/bash
#SBATCH --job-name=final_model_2d
#SBATCH --partition=gpu.stu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/final_model_2d_%j.txt

cd ~/project
source venv26/bin/activate
cd src/classification/ct_only

python final_model.py
