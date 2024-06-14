clear

CUDA_VISIBLE_DEVICES=7 python3 train.py yamls/finetune/llama2-7b_microadam_gsm8k.yaml \
        task=gsm8k \
        optimizer.name=microadam \
        optimizer.defaults.lr=4e-5 \
        optimizer.microadam.m=10 \
        optimizer.microadam.quant_block_size=64 \
        save_folder=./llama2_7b_gsm8k_microadam \
        seed=42