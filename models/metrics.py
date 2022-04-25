import torch
from torchmetrics import Metric


class LossDropAccumulator(Metric):
    def __init__(self) -> None:
        super().__init__(compute_on_step=True, dist_sync_on_step=False)
        self.add_state("drop_mask", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, drop_mask: torch.Tensor):
        self.drop_mask += (drop_mask == 0).int().sum()

    def compute(self):
        return self.drop_mask
