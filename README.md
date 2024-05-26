# MicroAdam
This repository contains the code to reproduce the results in the paper [MicroAdam: Accurate 
Adaptive Optimization with Low Space Overhead and Provable Convergence]().

## Installation
Follow the installation instruction from the 
[ISTA-DASLab-Optimizers](https://github.com/IST-DASLab/ISTA-DASLab-Optimizers?tab=readme-ov-file#installation) 
repository. You can choose whether you want a new environment or use your existing one.

In addition, we need to install the following packages:
```shell
pip3 install transformers bitsandbytes came-pytorch
```

## Usage
We provide code to reproduce the following experiments:
- BERT-Base/Large and OPT-1.3B using HuggingFace repository
- **[To Be Done]** Llama-2 7B / GLUE-MNLI using `llm-foundry` from MosaicML

Please use our code from this repo because we modified the original repositories to ease `wandb`
integration.

### Reproduce experiments for GLUE/MNLI
We provide the scripts `run_glue_X.sh`, where `X` is the optimizer name, as follows:
- `microadam`
- **[TODO]** `adamw`
- **[TODO]** `galore`
- **[TODO]** `came`
- **[TODO]** `adamw8b`

```shell
cd huggingface_glue_mnli
bash run_glue_microadam.sh
```