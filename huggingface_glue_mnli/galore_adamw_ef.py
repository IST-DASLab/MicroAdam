import wandb
import torch
import math
import time
import socket
from torch import topk as TopK
from torch.distributed import is_initialized, get_rank
from galore_projector import GaLoreProjector
from helpers import get_gpu_mem_usage, get_first_device, ModelBlockSplitter
import ista_daslab_micro_adam

class GaLoreAdamWEF(torch.optim.Optimizer):
    def __init__(self, params, lr, use_ef, quant_block_size, rank, svd_gap, scale=1, beta1=0.9, beta2=0.999, weight_decay=0, eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps)
        super(GaLoreAdamWEF, self).__init__(params, defaults)
        print(f'Using GaLoreAdamWEF optimizer')

        self.lr = lr
        self.use_ef = use_ef
        self.quant_block_size = quant_block_size
        self.rank = rank
        self.svd_gap = svd_gap
        self.scale = scale
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.steps = 0  # how many optimization steps were performed so far
        self.log_interval = 100
        self.device = get_first_device()

        self._init_state()
        print(f'[GaLoreAdamWEF] built optimizer')

        # self.printed = False

    def _init_state(self):
        print('[GaLoreAdamWEF] initializing layer states...')
        for group in self.param_groups:
            lr = group['lr']
            wd = group.get('weight_decay', self.weight_decay) # if the param groups do not have weight decay, then use the external one
            should_apply_galore = group['should_apply_galore']
            for p in group['params']:
                if not p.requires_grad:
                    continue

                layer_size = p.numel()
                st = self.state[p]

                st['lr'] = lr
                st['weight_decay'] = wd
                st['d'] = layer_size

                st['should_apply_galore'] = should_apply_galore # len(p.shape) > 1

                if should_apply_galore:
                    ### variables for error feedback: d_index_quant is the index where the last, smaller quantization block starts
                    if self.use_ef:
                        st['quant_full_blocks_count'], st['d_index_quant'] = ModelBlockSplitter.block_split(st['d'], self.quant_block_size)
                        st['error'] = torch.zeros(int(math.ceil(st['d'] / 2)), dtype=torch.uint8, device=self.device)
                        st['min_vals'] = torch.zeros(st['quant_full_blocks_count'] + 1, dtype=torch.bfloat16, device=self.device)
                        st['max_vals'] = torch.zeros(st['quant_full_blocks_count'] + 1, dtype=torch.bfloat16, device=self.device)

                    ### GaLore
                    st['galore_projector'] = GaLoreProjector(rank=self.rank, verbose=False, update_proj_gap=self.svd_gap, scale=self.scale)

                    ### adam states: will be initialized when we know the size of the r-dimensional gradient
                    st['m'] = None
                    st['v'] = None
                else:
                    ### adam states: initialize now for 1-dimensional tensors
                    st['m'] = torch.zeros_like(p)
                    st['v'] = torch.zeros_like(p)
        print('[GaLoreAdamWEF] initializing layer states...')

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1

        norm_g_d, norm_g_r, norm_u_d, norm_u_r, norm_e = 0, 0, 0, 0, 0
        self._update_lr_wd()

        # self.printed = False

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        time_start = time.time()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                ngd, ngr, nud, nur, ne = self.update_step(p) # norm_g_d, norm_g_r, norm_u_d, norm_u_r, norm_e
                norm_g_d += ngd
                norm_g_r += ngr
                norm_u_d += nud
                norm_u_r += nur
                norm_e += ne

        # torch.cuda.synchronize()
        time_end = time.time()
        elapsed_step = time_end - time_start
        self._log(norm_g_d, norm_g_r, norm_u_d, norm_u_r, norm_e, elapsed_step)
        # input('Press enter to continue...')
        return loss

    def nans(self, x):
        nans_percent = torch.isnan(x).sum() / x.numel() * 100.
        return f'{nans_percent:5.2f}%'

    @torch.no_grad()
    def update_step(self, p):
        norm_g_d, norm_g_r, norm_u_d, norm_u_r, norm_e = 0, 0, 0, 0, 0

        # bias correction terms for Adam
        bc1 = 1 - self.beta1 ** self.steps
        bc2 = math.sqrt(1 - self.beta2 ** self.steps)

        st = self.state[p]
        grad_d = p.grad
        grad_d_flat = grad_d.view(-1)

        if self.steps % self.log_interval == 0:
            norm_g_d = grad_d.norm(p=2) ** 2

        lr = st['lr']
        wd = st['weight_decay']
        d = st['d']
        should_apply_galore = st['should_apply_galore']

        # txt = f'{self.steps=} {str(list(p.shape)):16s}\tp={self.nans(p)} g_d={self.nans(grad_d)}'

        # if we set should_apply_galore to all parameters, then it will fail for 1-d tensors
        if should_apply_galore and len(grad_d.shape) > 1:
            projector = st['galore_projector']
            if self.use_ef:
                quant_full_blocks_count, d_index_quant = st['quant_full_blocks_count'], st['d_index_quant']

            ##### STEP 1: g^d += Qinv(xi)
            if self.use_ef:
                # acc = torch.zeros_like(grad_d)
                ista_daslab_micro_adam.asymm_block_quant_inv(d, self.quant_block_size, st['error'], st['min_vals'], st['max_vals'], grad_d)
                # txt = f'{txt} |a|={acc.norm(p=2)}'
                # grad_d.add_(acc)

            ##### STEP 2: GaLore projection
            grad_r = projector.project(grad_d, iter=self.steps)
            if self.steps % self.log_interval == 0:
                norm_g_r = grad_r.norm(p=2) ** 2

            # txt = f'{txt} g_r={self.nans(grad_r)} o_m={self.nans(projector.ortho_matrix)}'

            ##### STEP 3: GaLore back projection
            grad_d_back_proj = projector.project_back(grad_r)
            
            # txt = f'{txt} g_d_bp={self.nans(grad_d_back_proj)}'

            ##### STEP 4: grad_d will store the error now because we subtract grad_d_back_proj from grad_d (the difference is the error)
            grad_d.sub_(grad_d_back_proj)

            # txt = f'{txt} |xi|={grad_d.norm(p=2)}'

            ##### STEP 5: update min/max and then quantize the error (stored in g_d)
            if self.use_ef:
                if quant_full_blocks_count == 1:
                    st['min_vals'][:quant_full_blocks_count] = grad_d_flat[:d_index_quant].min()
                    st['max_vals'][:quant_full_blocks_count] = grad_d_flat[:d_index_quant].max()
                else:
                    st['min_vals'][:quant_full_blocks_count] = grad_d_flat[:d_index_quant].view(quant_full_blocks_count, self.quant_block_size).min(dim=1).values
                    st['max_vals'][:quant_full_blocks_count] = grad_d_flat[:d_index_quant].view(quant_full_blocks_count, self.quant_block_size).max(dim=1).values
                if d_index_quant < d:
                    st['min_vals'][quant_full_blocks_count] = grad_d_flat[d_index_quant:].min()
                    st['max_vals'][quant_full_blocks_count] = grad_d_flat[d_index_quant:].max()
                    
                ista_daslab_micro_adam.asymm_block_quant(d, self.quant_block_size, st['error'], st['min_vals'], st['max_vals'], grad_d_flat)  # error = Q(a, min, max)
                # txt = f'{txt} |e2|={st["error"].norm(p=2)}'

            ##### STEP 6:
            if st['m'] is None and st['v'] is None:
                st['m'] = torch.zeros_like(grad_r)
                st['v'] = torch.zeros_like(grad_r)

            st['m'].mul_(self.beta1).add_(grad_r, alpha=1-self.beta1)
            st['v'].mul_(self.beta2).addcmul_(grad_r, grad_r, value=1-self.beta2)
            ur = st['m'] / (st['v'].sqrt() + self.eps * bc2)

            # txt = f'{txt} u_r={self.nans(ur)}'
            if self.steps % self.log_interval == 0:
                norm_u_r = ur.norm(p=2) ** 2

            ##### STEP 7: project adam update from r-dimensional space to d-dimensional space
            p.grad.zero_().add_(projector.project_back(ur))
            if self.steps % self.log_interval == 0:
                norm_u_d = p.grad.norm(p=2) ** 2

            # txt = f'{txt} u_d={self.nans(p.grad)}'

            ##### STEP 8: update model
            p.mul_(1 - lr * wd).add_(p.grad, alpha=-lr * bc2 / bc1)
        else:
            if st['m'] is None and st['v'] is None:
                st['m'] = torch.zeros_like(grad_d)
                st['v'] = torch.zeros_like(grad_d)
            # do original AdamW for the one-dimensional parameters
            st['m'].mul_(self.beta1).add_(grad_d, alpha=1-self.beta1)
            st['v'].mul_(self.beta2).addcmul_(grad_d, grad_d, value=1 - self.beta2)

            p.grad = st['m'] / (st['v'].sqrt() + self.eps * bc2)

            # txt = f'{txt} u_d={self.nans(p.grad)}'

            p.mul_(1 - lr * wd).add_(p.grad, alpha=-lr * bc2 / bc1)

        # print(txt)

        if self.steps % self.log_interval == 0:
            if should_apply_galore and self.use_ef:
                grad_d.zero_()
                ista_daslab_micro_adam.asymm_block_quant_inv(d, self.quant_block_size, st['error'], st['min_vals'], st['max_vals'], grad_d)
                norm_e = grad_d.norm(p=2) ** 2

        return norm_g_d, norm_g_r, norm_u_d, norm_u_r, norm_e

    def _log(self, norm_g_d, norm_g_r, norm_u_d, norm_u_r, norm_e, elapsed_step):
        if self.steps % self.log_interval == 0:
            wandb_data = dict(
                optimizer_steps=self.steps,
                gpu_mem_usage=get_gpu_mem_usage(),
                norm_g_d=math.sqrt(norm_g_d),
                norm_g_r=math.sqrt(norm_g_r),
                norm_u_d=math.sqrt(norm_u_d),
                norm_u_r=math.sqrt(norm_u_r),
                norm_error=math.sqrt(norm_e),
                elapsed_step=elapsed_step
            )

            if not is_initialized() or get_rank() == 0:
                wandb.log(wandb_data, commit=False)
                print(wandb_data)

    def _update_lr_wd(self):
        # copy the learning rate group to parameter state because the lr scheduler updates the one in the group
        for group in self.param_groups:
            lr = group['lr']
            wd = group.get('weight_decay', self.weight_decay)  # if the param groups do not have weight decay, then use the external one
            for p in group['params']:
                self.state[p]['lr'] = lr
                self.state[p]['wd'] = wd
