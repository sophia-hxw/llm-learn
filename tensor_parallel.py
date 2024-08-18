import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, distribute_tensor, distribute_module

def tensor_parallel():
    device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])
    # if we want to do row-wise sharding
    rowwise_placement = [Shard(0)]
    # if we want to do col-wise sharding
    colwise_placement = [Shard(1)]

    big_tensor = torch.randn(888, 12)
    # returned will be sharded across the dimension specified in placements
    rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=rowwise_placement)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8,8)
        self.fc2 = nn.Linear(8,8)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.fc1(input)) + self.fc2(input)

def shard_params(mod_name, mod, mesh):
    rowwise_placement = [Shard(0)]
    def to_dist_tensor(t): return distribute_tensor(t, mesh, rowwise_placement)
    mod._apply(to_dist_tensor)

def shard_fc(mod_name, mod, mesh):
    rowwise_placement = [Shard(0)]
    if mod_name == "fc1":
        mod.weight = torch.nn.Parameter(distribute_tensor(mod.weight, mesh, rowwise_placement))

def module_parallel():
    mesh = DeviceMesh(device_type="cuda", mesh=[[0,1],[2,3]])
    shard_module = distribute_module(MyModule(), mesh, partition_fn=shard_params)
    shard_module = distribute_module(MyModule(), mesh, partition_fn=shard_fc)