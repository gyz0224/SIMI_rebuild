"""保存已完成批次的位置，避免 DataLoader 预取提前推进训练进度。"""

import math

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler):
    """按确定的样本顺序生成 batch，并由训练器确认已完成的 batch。

    工作进程可以提前请求后续批次，但只有 commit 才推进持久化游标。
    因此中途保存不会跳过已经预取但还没有完成更新的样本。
    """

    def __init__(self, size: int, batch_size: int, shuffle: bool, drop_last: bool, seed):
        self.size = size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = torch.Generator()
        self.generator.manual_seed(torch.initial_seed() if seed is None else seed)
        self.epoch = 0
        self.cursor = 0
        self.order = self._new_order()

    @property
    def total_batches(self) -> int:
        if self.drop_last:
            return self.size // self.batch_size
        return math.ceil(self.size / self.batch_size)

    def _new_order(self) -> torch.Tensor:
        if self.shuffle:
            return torch.randperm(self.size, generator=self.generator)
        return torch.arange(self.size)

    def __iter__(self):
        # 固定本次迭代的起点；预取本身不修改 cursor。
        for batch in range(self.cursor, self.total_batches):
            begin = batch * self.batch_size
            yield self.order[begin:begin + self.batch_size].tolist()

    def __len__(self):
        return self.total_batches - self.cursor

    def commit(self) -> None:
        if self.cursor >= self.total_batches:
            raise RuntimeError("当前 epoch 没有可提交的 batch")
        self.cursor += 1

    def next_epoch(self) -> None:
        if self.cursor != self.total_batches:
            raise RuntimeError("尚未完成当前 epoch，不能推进到下一轮")
        self.epoch += 1
        self.cursor = 0
        self.order = self._new_order()

    def state_dict(self) -> dict:
        return {
            "size": self.size, "batch_size": self.batch_size,
            "shuffle": self.shuffle, "drop_last": self.drop_last,
            "epoch": self.epoch, "cursor": self.cursor,
            "order": self.order.clone(), "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("size", "batch_size", "shuffle", "drop_last"):
            if state[key] != getattr(self, key):
                raise ValueError(f"完整恢复的 sampler 配置不匹配：{key}")
        order = state["order"].cpu()
        if (order.shape != (self.size,) or order.dtype != torch.int64
                or not torch.equal(order.sort().values, torch.arange(self.size))):
            raise ValueError("检查点中的样本排列不完整或不合法")
        if not 0 <= state["cursor"] <= self.total_batches:
            raise ValueError("检查点中的 batch 游标超出范围")
        if isinstance(state["epoch"], bool) or not isinstance(state["epoch"], int) or state["epoch"] < 0:
            raise ValueError("检查点中的 epoch 必须是非负整数")
        self.epoch = state["epoch"]
        self.cursor = state["cursor"]
        self.order = order.clone()
        self.generator.set_state(state["generator"].cpu())
