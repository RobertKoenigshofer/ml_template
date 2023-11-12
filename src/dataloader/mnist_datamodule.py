import os

import pytorch_lightning as pl
from torchvision.datasets import MNIST

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms



DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../data'))
DEFAULT_MEAN = 0.1307
DEFAULT_SD = 0.3081 


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, 
        data_dir: str = DATA_PATH, 
        batch_size: int = 128,
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((DEFAULT_MEAN,), (DEFAULT_SD,))
        ]),
        target_transform = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform= transform
        self.target_transform = target_transform

        self.default_mean = DEFAULT_MEAN
        self.default_sd = DEFAULT_SD

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        self.mnist_test = MNIST(
            self.data_dir, 
            train=False, 
            transform=self.transform,
            target_transform=self.target_transform
        )
        self.mnist_predict = MNIST(
            self.data_dir, 
            train=False, 
            transform=self.transform,
            target_transform=self.target_transform
        )
        mnist_full = MNIST(
            self.data_dir, 
            train=True, 
            transform=self.transform,
            target_transform= self.target_transform
        )
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()
        )