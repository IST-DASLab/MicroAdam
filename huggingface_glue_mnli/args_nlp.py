from dataclasses import dataclass, field


@dataclass
class CustomArgs:

    galore_use_ef: int = field(default=0, metadata={"help": "Whether to use EF for GaLore or not"})
    galore_rank: int = field(default=0, metadata={"help": "Rank for GaLore"})
    galore_svd_gap: int = field(default=0, metadata={"help": "SVD update gap for GaLore"})

    exclude_layers: str = field(default=None, metadata={"help": "Comma separated strigs representing the layers to be excluded from the optimizer"})
    use_bf16: int = field(default=1, metadata={"help": "Whether to use bf16 for values in sparse preconditioned methods or not"})
    lr: float = field(default=1e-4, metadata={"help": "The learning rate"})
    damp: float = field(default=1e-6, metadata={"help": "The dampening "})
    ngrads: int = field(default=1024, metadata={"help": "Number of gradients"})
    momentum: float = field(default=0, metadata={"help": "Momentum parameter"})
    k: float = field(default=1, metadata={"help": "The value to be used in Top-K"})


    optimizer_name: str = field(default="", metadata={"help": "Use optimizer name"})
    wandb_project: str = field(default="", metadata={"help": "the name for wandb project"})
    wandb_group: str = field(default="", metadata={"help": "the name for wandb group"})
    wandb_job_type: str = field(default="", metadata={"help": "the name for wandb job type"})
    wandb_name: str = field(default="", metadata={"help": "the name for wandb run name"})

    quant_block_size: int = field(default=64, metadata={"help": "Block size for quantization"})
    beta1: float = field(default=0.9, metadata={"help": "beta1 for Adam"})
    beta2: float = field(default=0.999, metadata={"help": "beta2 for Adam"})
    delta: float = field(default=0, metadata={"help": "delta for AdaBlockW"})
    eps: float = field(default=1e-8, metadata={"help": "eps for AdaBlockW and (Block)Adam"})
    block_size: int = field(default=8, metadata={"help": "block size for BlockAdam"})

