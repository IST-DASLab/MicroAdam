clear
MODEL=bert-base # bert-base OR bert-large OR opt-1.3b
OUTPUT_DIR=./results_hf_glue_mnli

WANDB_PROJECT=microadam
WANDB_GROUP=huggingface
WANDB_JOB_TYPE=glue_mnli
WANDB_NAME=test

SEED=42
LR=4e-5 # use --lr to set learning rate and let learning_rate set to 1e-4 (the last one will be ignored)
QUANT_BLOCK_SIZE=100000
NGRADS=10
DENSITY=0.01 # percentage, 0.01 means 1%

CUDA_VISIBLE_DEVICES=0 python glue.py \
    --num_train_epochs 3 \
    --optimizer_name microadam \
    --logging_strategy steps \
    --logging_steps 100 \
    --task_name mnli \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 0.0001 \
    --overwrite_output_dir \
    --save_strategy epoch \
    --save_total_limit 1 \
    --bf16 \
    --bf16_full_eval \
    \
    --model_name_or_path ${MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_group ${WANDB_GROUP} \
    --wandb_job_type ${WANDB_JOB_TYPE} \
    --wandb_name ${WANDB_NAME} \
    \
    --seed ${SEED} \
    --lr ${LR} \
    --quant_block_size ${QUANT_BLOCK_SIZE} \
    --ngrads ${NGRADS} \
    --k ${DENSITY} \
    \
    --weight_decay 0 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8