import os
import torch
import wandb
import gpustat
from importlib import import_module
from torch.distributed import is_initialized, get_rank

def get_cuda_capability(device=0):
    cc = torch.cuda.get_device_capability(device)
    cc_int = cc[0] * 10 + cc[1]
    return cc_int

def import_correct_cuda_cadam():
    try:
        cc = get_cuda_capability()
        module = import_module(f'cuda_cadam_sm{cc}')
        return module
    except ModuleNotFoundError as e:
        print(e)
        raise RuntimeError(f'The cuda_cadam library was not compiled for sm{cc}!')

def get_gpu_mem_usage():
    gpus = gpustat.new_query().gpus
    gids = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    gpu_mem = sum([int(proc['gpu_memory_usage']) for gid in gids for proc in gpus[gid]['processes'] if int(proc['pid']) == os.getpid()])
    return gpu_mem

def get_first_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if is_initialized():
        return torch.device(f'cuda:{get_rank()}')
    return torch.device('cuda:0')

def setup_wandb(project, job_type, group, name, config):
    return wandb.init(
        project=project,
        job_type=job_type,
        group=group,
        name=name,
        config=config,
        settings=wandb.Settings(start_method='fork'))

class ModelBlockSplitter:
    """
        This class contains methods that split a tensor of size d (model size) into different blocks
    to be used for Top-K or for quantizing Error Feedback.
        Examples:
            - block: returns pairs of indices of size block_size
    """
    @staticmethod
    def block_split(model_size, block_size):
        if model_size < block_size:
            return 1, model_size
        ### this is the shorter version that only returns the number of full blocks of size "block_size"
        ### and the starting position of the last and smallest block
        blocks_count = int(model_size / block_size)
        start_index_last_block = model_size - model_size % block_size
        return blocks_count, start_index_last_block