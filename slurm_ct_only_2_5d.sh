#!/bin/bash
#SBATCH --job-name=ct_only_2_5d
#SBATCH --partition=gpu.stu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/ct_only_2_5d_%j.txt

SEED=$1
shift

cd ~/project
source venv26/bin/activate
cd src/classification/ct_only

python train.py \
--mode 2.5d \
--epochs 100 \
--batch_size 16 \
--lr 1e-5 \
--weight_decay 5e-4 \
--freeze_until layer3_layer4 \
--patience 30 \
--seed $SEED \
"$@"
