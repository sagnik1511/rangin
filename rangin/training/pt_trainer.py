import gc
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict
from torch.utils.data import Dataset, DataLoader, random_split
from rangin.utils.training import *
import warnings
warnings.filterwarnings("ignore")


class RanginTrainer:

    def __init__(self, dataset: Dataset, model: nn.Module, arg_obj: TrainArgs):
        self.dataset = dataset
        self.model = model
        self.args = arg_obj
        self.res_df_train: pd.DataFrame = self.create_res_df(self.args.metrics)
        self.epoch_res_df_train: pd.DataFrame  = self.res_df_train.copy()
        self.res_df_val: pd.DataFrame  = self.res_df_train.copy()
        self.epoch_res_df_val: pd.DataFrame  = self.res_df_val.copy()
        self.metrics: Dict[str, float] = self.generate_metric_dict(self.args.metrics)
        self.criterion = self.generate_loss_function(self.args.loss)
        self.best_model_weights = self.model.state_dict()
        self.tr_portion = fetch_split_size(self.args.tv_split_ratio)
        self.optim: torch.optim = self.generate_optimizer()
        

    def run_single_batch(self, batch, training = True):
        x, y = batch
        if self.args.device.startswith("cuda"):
            x = x.cuda()
            y = y.cuda()
        op = self.model(x)
        batch_results = calculate_batch_results(op, y, self.metrics)
        loss = self.criterion(op, y)
        batch_results["loss"] = loss.item()
        self.epoch_res_df_train = self.epoch_res_df_train.append(batch_results, ignore_index=True)
        if training:
            loss.backward()
            self.optim.step()
        del x, y, op
        gc.collect()

    def train_single_epoch(self, train_loader, epoch):
        self.model.train()
        self.epoch_res_df_train = self.epoch_res_df_train.drop(self.epoch_res_df_train.index)
        for index, batch in enumerate(train_loader):
            self.run_single_batch(batch, training=True)
            if index % self.args.log_index == 0:
                print(self.epoch_res_df_train.tail(1))
        self.epoch_res_df_train["epoch"] = [epoch for _ in range(len(self.epoch_res_df_train))]
        avg_res = self.epoch_res_df_train.mean()
        avg_res_dict = {col : val for col, val in zip(avg_res.index, avg_res.values)}
        print(f"Training Scores :\n{avg_res_dict}")
        self.res_df = pd.concat([self.res_df_train, self.epoch_res_df_train], axis=1)

    def validate_single_epoch(self, val_loader, epoch):
        self.model.eval()
        self.epoch_res_df_val = self.epoch_res_df_val.drop(self.epoch_res_df_val.index)
        for batch in val_loader:
            self.run_single_batch(batch, training=False)
        self.epoch_res_df_val["epoch"] = [epoch for _ in range(len(self.epoch_res_df_val))]
        avg_res = self.epoch_res_df_val.mean()
        avg_res_dict = {col : val for col, val in zip(avg_res.index, avg_res.values)}
        print(f"Validation Scores :\n{avg_res_dict}")
        self.res_df = pd.concat([self.res_df_val, self.epoch_res_df_val], axis=1)

    def run_single_epoch(self, training_loader, validation_loader, epoch):
        print(f"Epoch : {epoch}")
        self.train_single_epoch(training_loader, epoch)
        self.validate_single_epoch(validation_loader, epoch)

    @staticmethod
    def create_res_df(metrics):
        empty_df = pd.DataFrame(columns=metrics+["loss"])
        return empty_df

    @staticmethod
    def generate_metric_dict(metrics):
        metric_dict = {}
        for metric_name in metrics:
            if metric_name == "mse":
                metric = nn.MSELoss()
            elif metric_name == "L1":
                metric = nn.L1Loss()
            else:
                raise NotImplementedError
            metric_dict[metric_name] = metric
        return metric_dict
    
    def generate_optimizer(self):
        if self.args.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError

    @staticmethod
    def generate_loss_function(lossf):
        if lossf == "mse":
            return nn.MSELoss()
        else:
            raise NotImplementedError

    def train(self):
        train_size = int(self.tr_portion * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_ds, val_ds = random_split(self.dataset, [train_size, val_size])
        train_dl = DataLoader(train_ds, batch_size=self.args.batch_size,
                    shuffle=self.args.shuffle, drop_last=self.args.drop_last)
        val_dl = DataLoader(val_ds, batch_size=self.args.batch_size,
                    shuffle=self.args.shuffle, drop_last=self.args.drop_last)
        if self.args.device.startswith("cuda"):
            self.model = self.model.cuda()
        for epoch in range(self.args.max_epochs):
            self.run_single_epoch(train_dl, val_dl, epoch+1)

