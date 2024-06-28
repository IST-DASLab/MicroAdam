clear

# To avoid package conflicts, we chose not to install llm-foundry using `pip3 install -e ".[gpu]"
# Instead, we are installing packages manually based on their version in the environment
# In the script MicroAdam/llm-foundry/scripts/train/train.py we manually append the path for llmfoundry package to sys.path,
# as well as the path for lm-evaluation-harness. Both MicroAdam and lm-evaluation-harness MUST be in the folder PROJECTS_ROOT
# Replace this with the folder where you have both MicroAdam and lm-evaluation-harness
# Here, we suppose that these folders are present in the home directory denoted by ~
export PROJECTS_ROOT=$(realpath ~)

CUDA_VISIBLE_DEVICES=0,1,2,3 composer train.py yamls/finetune/llama2-7b_microadam_gsm8k.yaml \
        task=gsm8k \
        optimizer.name=microadam \
        optimizer.defaults.lr=4e-5 \
        optimizer.microadam.m=10 \
        optimizer.microadam.quant_block_size=64 \
        save_folder=./llama2_7b_gsm8k_microadam \
        seed=42