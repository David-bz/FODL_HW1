import torch
import torch.utils.data.sampler as sampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
import torchvision.transforms as tvtf

class HW1_Dataset:
    def __init__(self, validation_portion = 0., batch_size = 1, subset_portion = 0.1, pca = None, mode='reg'):
        assert 0 < subset_portion <= 1.
        self.subset_portion = subset_portion
        self.batch_size = batch_size
        self.data_dir = os.path.expanduser('~/.pytorch-datasets')
        self.ds_train = torchvision.datasets.CIFAR10(root=self.data_dir, download=True, train=True, transform=tvtf.ToTensor())
        self.ds_test = torchvision.datasets.CIFAR10(root=self.data_dir, download=True, train=False, transform=tvtf.ToTensor())
        self.train_indices = np.random.choice(len(self.ds_train), int(self.subset_portion * len(self.ds_train)))
        self.test_indices = np.random.choice(len(self.ds_test), int(self.subset_portion * len(self.ds_test)))
        if mode == 'random_train':
            alternative_data = np.random.randint(0, 256, size=(len(self.ds_train),32,32,3),dtype=np.uint8)
            self.ds_train.data = alternative_data
        elif mode == 'half_random':
            indices = np.random.choice(len(self.ds_train), len(self.ds_train) // 2)
            alternative_data = np.random.randint(0, 256, size=(len(self.ds_train),32,32,3),dtype=np.uint8)
            self.ds_train.data[indices] = alternative_data[indices]
        elif mode == 'adversarial':
            indices = np.random.choice(len(self.ds_train), len(self.ds_train) // 2)
            for idx in indices:
                self.ds_train.targets[idx] = (self.ds_train.targets[idx] + 1) % 10

        if validation_portion > 0:
            self._init_with_validation(validation_portion)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.test_indices)
        self.dl_train = DataLoader(self.ds_train , batch_size=batch_size, sampler=train_sampler)
        self.dl_test = DataLoader(self.ds_test , batch_size=batch_size, sampler=test_sampler)

    def _init_with_validation(self, validation_portion):
        split = int(validation_portion * self.subset_portion * len(self.ds_train))
        self.train_indices = self.train_indices[split:]
        self.val_indices = self.train_indices[:split]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.val_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.test_indices)

        self.dl_train = DataLoader(self.ds_train , batch_size=batch_size, sampler=train_sampler)
        self.dl_valid = DataLoader(self.ds_train , batch_size=batch_size, sampler=val_sampler)
        self.dl_test = DataLoader(self.ds_test , batch_size=batch_size, sampler=test_sampler)

    def flatten(self, dataloader: DataLoader):
        # Combines batches from a DataLoader into a single tensor.
        out_tensors_cache = []
        for batch in dataloader:
            # Handle case of batch being a tensor (no labels)
            if torch.is_tensor(batch):
                batch = (batch,)
            # Handle case of batch being a dict
            elif isinstance(batch, dict):
                batch = tuple(batch[k] for k in sorted(batch.keys()))
            elif not isinstance(batch, tuple) and not isinstance(batch, list):
                raise TypeError("Unexpected type of batch object")

            for i, tensor in enumerate(batch):
                if i >= len(out_tensors_cache):
                    out_tensors_cache.append([])

                out_tensors_cache[i].append(tensor)

        out_tensors = tuple(
            # 0 is batch dimension
            torch.cat(tensors_list, dim=0) for tensors_list in out_tensors_cache
        )

        return out_tensors

    def transform_for_baseline(self):
        self.X_train, self.y_train = self.flatten(self.dl_train)
        X_validation, y_validation = self.flatten(self.dl_valid)
        self.X_train = torch.cat((self.X_train, X_validation))
        self.y_train = torch.cat((self.y_train, y_validation))
        self.X_test, self.y_test = self.flatten(self.dl_test)
        self.X_train = self.X_train.reshape(5000, 3 * 32 * 32)
        self.X_test = self.X_test.reshape(1000, 3 * 32 * 32)

if __name__ == '__main__':
    data = HW1_Dataset()