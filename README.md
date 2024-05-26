# MicroAdam
This repository contains the code to reproduce the results for the paper [MicroAdam: Accurate 
Adaptive Optimization with Low Space Overhead and Provable Convergence]().

## Installation
Follow the installation instructions from the 
[ISTA-DASLab-Optimizers](https://github.com/IST-DASLab/ISTA-DASLab-Optimizers?tab=readme-ov-file#installation) 
repository. You can choose whether you want a new environment or use your existing one.

In addition, we need to install the following packages:
```shell
pip3 install transformers bitsandbytes came-pytorch
```

## Usage
We provide code to reproduce the following experiments:
- BERT-Base/Large and OPT-1.3B using HuggingFace repository
- **[TODO]** Llama-2 7B / GLUE-MNLI using `llm-foundry` from MosaicML

Please use our code from this repo because we modified the original repositories to ease `wandb`
integration.

### Reproduce experiments for GLUE/MNLI
We provide the scripts `run_hf_glue_mnli_OPTIM.sh`, where `OPTIM` is the optimizer name, as follows: 
`microadam`, `adamw`, `galore`, `came`, `adamw8b`.

```shell
cd huggingface_glue_mnli
OPTIM=microadam
bash run_hf_glue_mnli_${OPTIM}.sh
```