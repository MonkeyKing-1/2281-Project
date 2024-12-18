#!/bin/bash
#SBATCH --job-name=learner-distill-pythia-datagen
#SBATCH --account=kempner_emalach_lab
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project/logs/%j/logs.out
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
#SBATCH --array=1-1

#load modules
module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01

#activate conda environment
source ~/.bashrc
conda activate 2281-project-env

#project directory
cd /n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project

#environment variables
#export SPECIAL_TOKEN="<|sep|>"

#run the script

TARGET_MODEL="EleutherAI/pythia-2.8b"
METRIC="lk"
BATCH_SIZE=4
EPOCHS=1
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
DATASET_SPLIT="train"
MAX_EXAMPLES=50000
SIZES="70 160 410 70 160 410 70 160 410 70 160 410"

DRAFTERS_TEMP1=(
    distillation_lambada/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-70m_cimec/lambada_temperature_1.0_2024-12-17_19-02-08
    distillation_lambada/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-160m_cimec/lambada_temperature_1.0_2024-12-17_18-42-11
    distillation_lambada/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-410m_cimec/lambada_temperature_1.0_2024-12-17_18-23-04
    distillation_openwebtext_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-70m_stas/openwebtext-10k_temperature_1.0_2024-12-17_19-01-51
    distillation_openwebtext_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-160m_stas/openwebtext-10k_temperature_1.0_2024-12-17_18-39-17
    distillation_openwebtext_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-410m_stas/openwebtext-10k_temperature_1.0_2024-12-17_18-23-03
    distillation_pile_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-70m_NeelNanda/pile-10k_temperature_1.0_2024-12-17_19-03-16
    distillation_pile_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-160m_NeelNanda/pile-10k_temperature_1.0_2024-12-17_18-48-44
    distillation_pile_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-410m_NeelNanda/pile-10k_temperature_1.0_2024-12-17_18-23-10
    distillation_redpajama_v2/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-70m_vilm/RedPajama-v2-small_temperature_1.0_2024-12-17_19-07-08
    distillation_redpajama_v2/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-160m_vilm/RedPajama-v2-small_temperature_1.0_2024-12-17_18-50-18
    distillation_redpajama_v2/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-410m_vilm/RedPajama-v2-small_temperature_1.0_2024-12-17_18-23-11
)

DRAFTERS_TEMP2=(
    distillation_lambada/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-70m_cimec/lambada_temperature_2.0_2024-12-17_18-20-56
    distillation_lambada/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-160m_cimec/lambada_temperature_2.0_2024-12-17_18-21-06
    distillation_lambada/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-410m_cimec/lambada_temperature_2.0_2024-12-17_18-21-33
    distillation_openwebtext_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-70m_stas/openwebtext-10k_temperature_2.0_2024-12-17_18-20-46
    distillation_openwebtext_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-160m_stas/openwebtext-10k_temperature_2.0_2024-12-17_18-21-06
    distillation_openwebtext_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-410m_stas/openwebtext-10k_temperature_2.0_2024-12-17_18-21-33
    distillation_pile_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-70m_NeelNanda/pile-10k_temperature_2.0_2024-12-17_19-20-48
    distillation_pile_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-160m_NeelNanda/pile-10k_temperature_2.0_2024-12-17_18-21-06
    distillation_pile_10k/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-410m_NeelNanda/pile-10k_temperature_2.0_2024-12-17_18-21-33
    distillation_redpajama_v2/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-70m_vilm/RedPajama-v2-small_temperature_2.0_2024-12-17_18-21-35
    distillation_redpajama_v2/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-160m_vilm/RedPajama-v2-small_temperature_2.0_2024-12-17_18-21-35
    distillation_redpajama_v2/EleutherAI_pythia-2.8b_to_EleutherAI_pythia-410m_vilm/RedPajama-v2-small_temperature_2.0_2024-12-17_18-21-35
)
#temperature 1
# python main.py \
#     --mode create_dataset \
#     --target_model_name "$TARGET_MODEL" \
#     --drafters "${DRAFTERS_TEMP1[@]}" \
#     --sizes $SIZES \
#     --metric $METRIC \
#     --ptfile pythia_distilled_allDrafters_lk-epochs${EPOCHS}-batch${BATCH_SIZE}_temp1.0.pt \
#     --batch_size $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --dataset_name $DATASET_NAME \
#     --dataset_config $DATASET_CONFIG \
#     --dataset_split $DATASET_SPLIT \
#     --max_examples $MAX_EXAMPLES

#temperature 2
python main.py \
    --mode create_dataset \
    --target_model_name "$TARGET_MODEL" \
    --drafters "${DRAFTERS_TEMP2[@]}" \
    --sizes $SIZES \
    --metric $METRIC \
    --ptfile pythia_distilled_allDrafters_lk-epochs${EPOCHS}-batch${BATCH_SIZE}_temp2.0.pt \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --dataset_name $DATASET_NAME \
    --dataset_config $DATASET_CONFIG \
    --dataset_split $DATASET_SPLIT \
    --max_examples $MAX_EXAMPLES

# python main.py --mode train_learner \
#     --target_model_name EleutherAI/pythia-2.8b \
#     --ptfile data/pythia-lk-epochs20-batch4.pt \
#     --drafters_idx 0 1 2 \
#     --metric=lk \
#     --lk_k=1 \
#     --epochs=20 \
#     --hidden_dim=32 \
#     --num_layers=3 \
#     --dropout=0.3 \
#     --sizes 7 16 41

# python main.py --mode train_learner \
#     --target_model_name EleutherAI/pythia-2.8b \
#     --ptfile data/pythia-kl-epochs20-batch4.pt \
#     --drafters_idx 0 1 2 \
#     --metric=l2 \
#     --epochs=20 \
#     --hidden_dim=256 \
#     --num_layers=25 \
#     --dropout=0.3 \
#     --sizes 7 16 41

# python main.py --mode train_learner \
#     --target_model_name bigscience/bloomz-7b1 \
#     --ptfile data/bloom-kl-epochs20.pt \
#     --drafters_idx 0 1 2 3 \
#     --metric=kl \
#     --epochs=5 \
#     --hidden_dim=128 \
#     --num_layers=10 \
#     --dropout=0.3 \
#     --sizes 56 56 110 110

#python main.py --target_model_name bigscience/bloomz-7b1 --mode create_dataset --epochs 20 --batch_size 32 --drafters bigscience/bloomz-560m bigscience/bloom-560m bigscience/bloomz-1b1 bigscience/bloom-1b1 --metric 'lk' --ptfile 'bloom-lk-epochs20-batch32.pt' --sizes 56 56 110 110
#python main.py --target_model_name EleutherAI/pythia-2.8b --mode create_dataset --epochs 20 --batch_size 4 --drafters EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m  --metric 'kl' --ptfile 'pythia-kl-epochs20-batch4.pt' --sizes 7 16 41
