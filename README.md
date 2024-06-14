# MicroAdam
This repository contains the code to reproduce the results for the paper [MicroAdam: Accurate 
Adaptive Optimization with Low Space Overhead and Provable Convergence](https://arxiv.org/pdf/2405.15593).

## Installation
Follow the installation instructions from the 
[ISTA-DASLab-Optimizers](https://github.com/IST-DASLab/ISTA-DASLab-Optimizers?tab=readme-ov-file#installation) 
repository. You can choose whether you want a new environment or use your existing one.

### Summary:
```shell
pip3 install ista-daslab-optimizers
```

In addition, we need to install the following packages:
```shell
pip3 install transformers mosaicml bitsandbytes came-pytorch
```

## Usage
We provide code to reproduce the following experiments:
- BERT-Base/Large and OPT-1.3B on GLUE/MNLI using HuggingFace repository
- Llama-2 7B on GSM8k using [`llm-foundry` from MosaicML](https://github.com/mosaicml/llm-foundry)

Please use our code from this repo because we modified the original repositories to ease `wandb`
integration.

### Clone repository
```shell
cd ~ && git clone git@github.com:IST-DASLab/MicroAdam.git
```

### Reproduce experiments for GLUE/MNLI
We provide the scripts `run_hf_glue_mnli_OPTIM.sh`, where `OPTIM` is the optimizer name, as follows: 
`microadam`, `adamw`, `galore`, `came`, `adamw8b`.

```shell
cd ~/MicroAdam/huggingface_glue_mnli
# bash run_hf_glue_mnli_adamw.sh
# bash run_hf_glue_mnli_adamw8b.sh
# bash run_hf_glue_mnli_came.sh
# bash run_hf_glue_mnli_galore.sh
bash run_hf_glue_mnli_microadam.sh
```

### Reproduce experiments for Llama-2 7B on GSM-8k
We evaluate the model `lm-evaluation-harness` immediately after the training to log the results to wandb. We
need to install the evaluation package at the commit specified below:

```shell
cd ~
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
pip install -e .
```

Now we can run the experiments using the following commands, supposing that we are located in
`~/MicroAdam/llm-foundry/scripts/train`:

#### Run MicroAdam
```shell
cd ~/MicroAdam/llm-foundry/scripts/train
bash run_llama2-7b_gsm8k_microadam.sh
```

#### Run AdamW-8bit
```shell
python3 train.py yamls/finetune/llama2-7b_microadam_gsm8k.yaml \
        task=gsm8k \
        optimizer.name=adamw8b \
        optimizer.defaults.lr=5e-5 \
        save_folder=./llama2_7b_gsm8k_adamw8b \
        seed=42
```

#### Run DecoupledAdamW
```shell
python3 train.py yamls/finetune/llama2-7b_microadam_gsm8k.yaml \
        task=gsm8k \
        optimizer.name=decoupled_adamw \
        optimizer.defaults.lr=5e-5 \
        save_folder=./llama2_7b_gsm8k_decoupled_adamw \
        seed=42
```

## Changes compared to the original `llm-foundry` repository:
- [method `build_optimizer`](https://github.com/IST-DASLab/MicroAdam/blob/main/llm-foundry/llmfoundry/utils/builders.py#L373)
- changes in `llm-foundry/scripts/train/train.py:`
    * set `run_name` and `save_folder` depending on wandb group, job_type and name
    * added [evaluation](https://github.com/IST-DASLab/MicroAdam/blob/main/llm-foundry/scripts/train/evaluation.py) and [time elapsed](https://github.com/IST-DASLab/MicroAdam/blob/main/llm-foundry/scripts/train/train.py#L697) to be logged to wandb
    * added wandb_groups_config to [finetuning yaml](https://github.com/IST-DASLab/MicroAdam/blob/main/llm-foundry/scripts/train/yamls/finetune/llama2-7b_microadam_gsm8k.yaml#L105)
- changes in [finetuning yaml file](https://github.com/IST-DASLab/MicroAdam/blob/main/llm-foundry/scripts/train/yamls/finetune/llama2-7b_microadam_gsm8k.yaml):
    * added `task` variable
    * added `wandb_groups` section

## Citing
If you find our work useful, please consider citing:
```
@misc{modoranu2024microadam,
      title={MicroAdam: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence}, 
      author={Ionut-Vlad Modoranu and Mher Safaryan and Grigory Malinovsky and Eldar Kurtic and Thomas Robert and Peter Richtarik and Dan Alistarh},
      year={2024},
      eprint={2405.15593},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```