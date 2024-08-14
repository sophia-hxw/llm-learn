import torch
from torch.nn import nn

class llamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_status):
        input_dtype = hidden_status.dtype
        variance = hidden_status.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_status = hidden_status * torch.rsqrt(variance +self.variance_epsilon)
        return (self.weight * hidden_status).to(input_dtype)