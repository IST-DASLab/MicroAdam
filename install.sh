clear

conda create --name microadam python=3.9 -y
conda activate microadam

pip3 install cmake packaging # required for llm-foundry
pip3 install ista-daslab-optimizers
pip3 install transformers mosaicml mosaicml-streaming bitsandbytes came-pytorch

### We evaluate the model `lm-evaluation-harness` immediately after the training to log the results to wandb
### We need to install the evaluation package at the commit specified below:
cd ~
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
cd ~/lm-evaluation-harness
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
pip install -e .

cd ~/MicroAdam
pip3 install -e ".[gpu]"