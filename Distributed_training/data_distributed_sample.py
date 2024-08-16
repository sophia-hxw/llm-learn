from torch.utils.data import Sampler
import torch.distributed as dist
import math
import torch

class DistributedSample(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_availabel():
                raise RuntimeError("Requires distributed package to be avaiblable")
            num_replicas = dist.get_world_size()
        
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be avaiblable")
            rank - dist.get_rank()
        
        self.dataset = dataset # 数据集
        self.num_replicas = num_replicas # 进程个数，默认等于world_size，即GPU个数
        self.rank = rank # 当前属于哪个进程或者那块GPU
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 /self.num_replicas)) #每个进程的样本个数
        self.total_size = self.num_samples * self.num_replicas # 数据集的总样本数
        self.shuffle = shuffle # 是否打乱数据集
        self.seed = seed

    def __iter__(self):
        # shuffle处理：打乱数据集
        if self.shuffle:
            g = torch.Generator() #根据epoch和种子进行混淆
            # self.seed是一个固定值，通过set_epoch 改变self.epoch可以改变初始化种子
            # 所以可以让每一个epoch中数据集的打乱顺序不同
            # 使每一个epoch中，每一块GPU拿到的数据都不一样
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # 数据补充
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # 分配数据
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)#会返回一个迭代器，这个迭代器可以逐个访问 indices 中的每个元素。

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""给当前样本设定epoch值
        如果shuffle为True，那么所有进程的每个epoch都会有一个不同的随机顺序
        否则，当前样本的下一个迭代用的是相同的顺序
        输入的epoch参数是一个整数值
        """
        self.epoch = epoch

