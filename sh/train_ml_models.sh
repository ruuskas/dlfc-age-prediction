#!/bin/bash
#SBATCH --job-name=mlfc_age
#SBATCH --output=/scratch/nbe/restmeg/dlfc_age/slurm_logs/age_pred_ml-%A_%a.out
#SBATCH --error=/scratch/nbe/restmeg/dlfc_age/error_logs/errors_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7

PIPELINE_PATH=/scratch/nbe/restmeg/dlfc_age/dlfc_age_pipeline

ml purge
ml mamba

source activate agepred

cd "$PIPELINE_PATH" || exit 1
echo "Starting analysis"

metrics=("wpli" "aec")

job_id="$SLURM_ARRAY_TASK_ID"
metric_idx=$((job_id / 4))
remainder=$((job_id % 4))
use_graph_features=$((remainder / 2))
sensor_space=$((remainder % 2))

metric=${metrics[metric_idx]}

if [[ "$use_graph_features" -eq 0 ]]
then
  use_graph_features=""
else
  use_graph_features="--use-graph-features"
fi

if [[ "$sensor_space" -eq 0 ]]
then
  sensor_space=""
else
  sensor_space="--sensor-space"
fi

# Run the Python script with the task ID to select the hyperparameters
echo "Running model svr with metric $metric, $use_graph_features, $sensor_space"
srun python ml_02_train_models.py svr "$metric" $use_graph_features $sensor_space --use-dl-split|| exit 1

echo "Running model ridge with metric $metric, $use_graph_features, $sensor_space"
srun python ml_02_train_models.py ridge "$metric" $use_graph_features $sensor_space --use-dl-split || exit 1
echo "Completed analysis"
