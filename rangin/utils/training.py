import torch
from typing import List

def calculate_batch_results(pred, true, metrics):
    res_dict = {}
    for name, metric in metrics.items():
        res_dict[name] = metric(true, pred).cpu().detach().numpy()
    return res_dict

def fetch_split_size(ratio):
    return ratio / (ratio+1)

class TrainArgs:

    def __init__(self):

        # device
        self.device: str = "cuda:0"

        # dataset
        self.batch_size: int = 16
        self.shuffle: bool = True
        self.tv_split_ratio: float = 4
        self.image_shape: int = 224
        self.drop_last: bool = True
        
        # model

        # optimizer
        self.optimizer_name: str = "adam"
        self.lr: float = 1e-4
        self.weight_decay: float = 1e-6
        self.momentum: float = 0.9

        # metrics
        self.loss: str = "mse"
        self.metrics: List[str] = ["L1"]


        # runner
        self.max_epochs: int = 3
        self.log_index = 10

        # callbacks
