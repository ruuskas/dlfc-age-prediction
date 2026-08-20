#!/bin/bash
#SBATCH --job-name=dlfc_age
#SBATCH --output=/scratch/nbe/restmeg/dlfc_age/slurm_logs/age_prediction-%A_%a.out
#SBATCH --error=/scratch/nbe/restmeg/dlfc_age/error_logs/errors_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-53

PIPELINE_PATH=/scratch/nbe/restmeg/dlfc_age/dlfc_age_pipeline

ml purge
ml mamba

source activate agepred

cd "$PIPELINE_PATH" || exit 1
echo "Starting analysis"

models=("m1_depthwise" "m2_sym3d" "m3_conv3d")

job_id="$SLURM_ARRAY_TASK_ID"
model_idx=$((job_id / 18))
remainder=$((job_id % 18))
run_idx=$((remainder / 6))
dataset_idx=$((remainder % 6))
model_name=${models[model_idx]}


# Run the Python script with the task ID to select the hyperparameters
echo "Running model $model_name on dataset index $dataset_idx, run index $run_idx"
srun python dl_03_train_dl_models.py "$model_name" "$dataset_idx" "$run_idx" || exit 1
echo "Completed analysis"
