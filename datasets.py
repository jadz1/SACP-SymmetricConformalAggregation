import openml
from torchvision import datasets
import numpy as np
import torch


def load_datasets(name='FashionMNIST', opml=False):
    if opml:
        task = openml.tasks.get_task(name)
        features, targets = task.get_X_and_y(dataset_format='dataframe')
        X, y = features.values, targets.values
        return X, y
    elif name == 'MNIST':
        train = datasets.MNIST('./data', train=True, download=True)
        test = datasets.MNIST('./data', train=False, download=True)
        data = torch.cat([train.data.unsqueeze(1), test.data.unsqueeze(1)], 0).float() / 255.0
        y = np.concatenate([train.targets.numpy(), test.targets.numpy()])
        X = data.numpy()
    elif name=='CIFAR10':
        train = datasets.CIFAR10('./data', train=True, download=True)
        test = datasets.CIFAR10('./data', train=False, download=True)
        data_np = np.concatenate([train.data, test.data],0).astype(np.float32)/255.0
        y = np.concatenate([np.array(train.targets), np.array(test.targets)])
        X = np.moveaxis(data_np, -1, 1)
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
        X = (X - mean) / std
    else:
        raise ValueError(f"Unknown dataset or OpenML ID: {name}")
    return X, y



