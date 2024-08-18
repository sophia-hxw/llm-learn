import os
import torch
import torch.distributed
import torch.nn as nn
import torch.distributed.pipeline.sync.pipe as Pipe

# 初始化RPC框架
os.enversion['MASTER_ADDR'] = 'localhost'
os.enversion['MASTER_PORT'] = '29500'

torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

# 新建一个两层的线性层
fc1 = nn.Linear(16, 8).cuda(0)
fc2 = nn.Linear(8,4).cuda(1)

# 将两个线性层串起来
model = nn.Sequential(fc1, fc2)

# 构建流水线
model = Pipe(model, chunks=8)

# 训练 or 推理
input = torch.rand(16, 16).cuda(0)
output_ref = model(input)


