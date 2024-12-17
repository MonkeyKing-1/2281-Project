#!/bin/bash
#SBATCH --job-name=learner-distill-pythia
#SBATCH --account=kempner_emalach_lab
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project/logs-distillation/%j/logs.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=russell_li@college.harvard.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --mem=250GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
#SBATCH --exclude=holygpu8a19405
#SBATCH --array=1-4

module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01

source ~/.bashrc
conda activate 2281-project-env

nvidia-smi

cd /n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project

TARGET_MODEL="EleutherAI/pythia-2.8b"
STUDENT_MODEL="EleutherAI/pythia-70m"
EPOCHS=10
BATCH_SIZE=4
LR=1e-5
TEMP=1.0
SAVE_INTERVAL=50

DATASETS=(
    "stas/openwebtext-10k,,train,distillation_openwebtext_10k,50000,10,1.0"
    "cimec/lambada,,validation,distillation_lambada,50000,10,1.0"
    "NeelNanda/pile-10k,,train,distillation_pile_10k,50000,10,1.0"
    "vilm/RedPajama-v2-small,,train,distillation_redpajama_v2,50000,10,1.0"
)

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
ds="${DATASETS[$INDEX]}"

#The format for each dataset in the array is: dataset_name,dataset_config,dataset_split,distillation_directory,max_examples,epochs,temperature
IFS=',' read -r DATASET_NAME DATASET_CONFIG DATASET_SPLIT DISTILL_DIR MAX_EX EX_EPOCHS EX_TEMP <<< "$ds"
[ -z "$DATASET_SPLIT" ] && DATASET_SPLIT="train"
[ -z "$DISTILL_DIR" ] && DISTILL_DIR="distillation"
[ -z "$MAX_EX" ] && MAX_EX_ARG=() || MAX_EX_ARG=(--max_examples "$MAX_EX")
[ -z "$DATASET_CONFIG" ] && CONFIG_ARG=() || CONFIG_ARG=(--dataset_config "$DATASET_CONFIG")
[ -z "$EX_EPOCHS" ] && EX_EPOCHS=$EPOCHS
[ -z "$EX_TEMP" ] && EX_TEMP=$TEMP

echo "Distilling with dataset: $DATASET_NAME"
echo "Using dataset config: $DATASET_CONFIG"
echo "Split: $DATASET_SPLIT"
echo "Distillation directory: $DISTILL_DIR"
echo "Max examples: $MAX_EX"
echo "Epochs: $EX_EPOCHS"
echo "Temperature: $EX_TEMP"
echo "Save Interval: $SAVE_INTERVAL"
echo "-----------------------------------"

python main.py \
    --mode distill \
    --dataset_name "$DATASET_NAME" \
    "${CONFIG_ARG[@]}" \
    --dataset_split "$DATASET_SPLIT" \
    "${MAX_EX_ARG[@]}" \
    --distillation_directory "$DISTILL_DIR" \
    --target_model_name "$TARGET_MODEL" \
    --student_model_name "$STUDENT_MODEL" \
    --epochs "$EX_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr_distillation "$LR" \
    --temperature "$EX_TEMP" \
    --save_interval "$SAVE_INTERVAL"
