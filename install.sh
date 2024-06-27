clear

conda create --name microadam python=3.9 -y && conda activate microadam

pip3 install cmake packaging # required for llm-foundry

#cd ~/MicroAdam/llm-foundry && pip3 install -e ".[gpu]"

pip3 install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 torchmetrics==1.3.2 torch-optimizer==0.3.0
pip3 install accelerate==0.25.0 transformers==4.40.2 datasets==2.19.1 einops==0.7.0 triton==2.3.0 huggingface-hub==0.22.2
pip3 install mosaicml==0.22.0 mosaicml-cli==0.6.23 mosaicml-streaming==0.7.5

pip3 install ista-daslab-optimizers bitsandbytes came-pytorch # transformers mosaicml mosaicml-streaming

### We evaluate the model `lm-evaluation-harness` immediately after the training to log the results to wandb
### We need to install the evaluation package at the commit specified below:
cd ~ && git clone git@github.com:EleutherAI/lm-evaluation-harness.git && cd ~/lm-evaluation-harness && git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463 && pip3 install -e .