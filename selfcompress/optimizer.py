import torch
import torch.nn as nn


class DynamicAdam:
    def __init__(
            self,
            model: nn.Module,
            lr_weight=1e-3,
            lr_quantization=0.15,
            eps_weight = 1e-5,
            eps_quantization = 1e-3,
            beta_m=0.9,
            beta_v=0.999,
            weight_decay=5e-4
    ):
        self.model            = model
        self.lr_weight        = lr_weight
        self.lr_quantization  = lr_quantization
        self.eps_weight       = eps_weight
        self.eps_quantization = eps_quantization
        self.beta_m           = beta_m
        self.beta_v           = beta_v
        self.weight_decay     = weight_decay
        self.state = {}
        self.t     = 0
        self._init_states_()

    def _init_states_(self):
        for name, p in self.model.named_parameters():
            self.state[id(p)] = {
                'm': torch.zeros_like(p.data),
                'v': torch.zeros_like(p.data),
                'is_quant': ('bits' in name or 'exponent' in name),
                'is_bits': ('bits' in name),
                'is_weight': ('weight' in name and p.ndim >= 2)
            }

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def scale_learning_rate(self, factor: float):
        self.lr_weight       *= factor
        self.lr_quantization *= factor

    def cut_off_parameter(self, param_id, keep_indices: torch.Tensor, dim: int):
        old_state = self.state[param_id]
        self.state[param_id]['m'] = torch.index_select(input=old_state['m'], dim=dim, index=keep_indices)
        self.state[param_id]['v'] = torch.index_select(input=old_state['v'], dim=dim, index=keep_indices)

    @torch.no_grad()
    def step(self):
        self.t += 1
        beta_m, beta_v = self.beta_m, self.beta_v
        bcm = 1.0 - beta_m ** self.t
        bcv = 1.0 - beta_v ** self.t

        for p in self.model.parameters():
            if p.grad is None:
                continue

            grad = p.grad
            param_state = self.state[id(p)]

            if param_state["is_weight"] and self.weight_decay > 0:
                grad = grad.add(p.data, alpha=self.weight_decay)

            param_state['m'].mul_(beta_m).add_(grad, alpha=1.0 - beta_m)
            param_state['v'].mul_(beta_v).addcmul_(grad, grad, value=1.0 - beta_v)

            m_hat = param_state['m'] / bcm
            v_hat = param_state['v'] / bcv

            learning_rate = self.lr_quantization  if param_state['is_quant'] else self.lr_weight
            eps           = self.eps_quantization if param_state['is_quant'] else self.eps_weight

            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-learning_rate)
            
            if param_state['is_bits']:
                p.data.clamp_(min=0.0)


class DynamicAdamW:
    def __init__(
            self,
            model: nn.Module,
            lr_weight=1e-3,
            lr_quantization=0.15,
            eps_weight = 1e-5,
            eps_quantization = 1e-3,
            beta_m=0.9,
            beta_v=0.999,
            weight_decay=1e-2
    ):
        self.model            = model
        self.lr_weight        = lr_weight
        self.lr_quantization  = lr_quantization
        self.eps_weight       = eps_weight
        self.eps_quantization = eps_quantization
        self.beta_m           = beta_m
        self.beta_v           = beta_v
        self.weight_decay     = weight_decay
        self.state = {}
        self.t     = 0
        self._init_states_()

    def _init_states_(self):
        for name, p in self.model.named_parameters():
            self.state[id(p)] = {
                'm': torch.zeros_like(p.data),
                'v': torch.zeros_like(p.data),
                'is_quant': ('bits' in name or 'exponent' in name),
                'is_bits': ('bits' in name),
                'is_weight': ('weight' in name and p.ndim >= 2)
            }

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def scale_learning_rate(self, factor: float):
        self.lr_weight       *= factor
        self.lr_quantization *= factor

    def cut_off_parameter(self, param_id, keep_indices: torch.Tensor, dim: int):
        old_state = self.state[param_id]
        self.state[param_id]['m'] = torch.index_select(input=old_state['m'], dim=dim, index=keep_indices)
        self.state[param_id]['v'] = torch.index_select(input=old_state['v'], dim=dim, index=keep_indices)

    @torch.no_grad()
    def step(self):
        self.t += 1
        beta_m, beta_v = self.beta_m, self.beta_v
        bcm = 1.0 - beta_m ** self.t
        bcv = 1.0 - beta_v ** self.t

        for p in self.model.parameters():
            if p.grad is None:
                continue

            grad = p.grad
            param_state = self.state[id(p)]

            learning_rate = self.lr_quantization  if param_state['is_quant'] else self.lr_weight
            eps           = self.eps_quantization if param_state['is_quant'] else self.eps_weight

            # Decoupled weight decay (AdamW): shrink the weight directly, outside of m and v
            if param_state["is_weight"] and self.weight_decay > 0:
                p.data.mul_(1.0 - learning_rate * self.weight_decay)

            param_state['m'].mul_(beta_m).add_(grad, alpha=1.0 - beta_m)
            param_state['v'].mul_(beta_v).addcmul_(grad, grad, value=1.0 - beta_v)

            m_hat = param_state['m'] / bcm
            v_hat = param_state['v'] / bcv

            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-learning_rate)

            if param_state['is_bits']:
                p.data.clamp_(min=0.0)