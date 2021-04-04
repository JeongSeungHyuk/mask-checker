import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

import os
import csv
from PIL import Image


transform = transforms.Compose([
    #transforms.CenterCrop([400, 200]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MaskDataset(Dataset):
    config = {
        'paths': {
            'images_dir': '/opt/ml/input/data/train/images',
            'metadata': 'data/metadata.csv'
        }
    }


    def __init__(self, train=True):
        self.paths = self.config['paths']

        self.dataset = []
        with open(self.paths['metadata']) as f:
            reader = csv.reader(f)
            self.dataset = list(reader)[1:]

        num = int(len(self) * 0.80)
        if train:
            self.dataset = self.dataset[:num]
        else:
            self.dataset = self.dataset[num:]


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        data = self.dataset[idx]

        if data[0] == 'male':
            gender = 0
        else:
            gender = 1

        data[1] = int(data[1])
        if data[1] < 30:
            age = 0
        elif data[1] < 60:
            age = 1
        else:
            age = 2

        if data[2] == 'normal':
            mask = 2
        elif data[2] == 'incorrect_mask':
            mask = 1
        else:
            mask = 0

        labels = [mask, gender, age]
        if labels == [0, 0, 0]:
            label = 0
        elif labels == [0, 0, 1]:
            label = 1
        elif labels == [0, 0, 2]:
            label = 2
        elif labels == [0, 1, 0]:
            label = 3
        elif labels == [0, 1, 1]:
            label = 4
        elif labels == [0, 1, 2]:
            label = 5
        elif labels == [1, 0, 0]:
            label = 6
        elif labels == [1, 0, 1]:
            label = 7
        elif labels == [1, 0, 2]:
            label = 8
        elif labels == [1, 1, 0]:
            label = 9
        elif labels == [1, 1, 1]:
            label = 10
        elif labels == [1, 1, 2]:
            label = 11
        elif labels == [2, 0, 0]:
            label = 12
        elif labels == [2, 0, 1]:
            label = 13
        elif labels == [2, 0, 2]:
            label = 14
        elif labels == [2, 1, 0]:
            label = 15
        elif labels == [2, 1, 1]:
            label = 16
        elif labels == [2, 1, 2]:
            label = 17

        image = Image.open(f'{self.paths["images_dir"]}/{data[3]}')
        image = transform(image)

        return image, label


class MaskTestDataset(Dataset):
    config = {
        'paths': {
            'images_dir': '/opt/ml/input/data/eval/images',
            'metadata': '/opt/ml/input/data/eval/info.csv'
        }
    }

    def __init__(self, train=True):
        self.paths = self.config['paths']

        self.dataset = []
        with open(self.paths['metadata']) as f:
            reader = csv.reader(f)
            self.dataset = list(reader)[1:]


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        path = self.dataset[idx][0]
        image = Image.open(f'{self.paths["images_dir"]}/{path}')
        image = transform(image)

        return image, path


if __name__ == '__main__':
    train_data = MaskDataset(train=True)
    print(len(train_data))
    for i in range(1):
        print(train_data[i])

    test_data = MaskTestDataset(train=True)
    print(len(test_data))
    for i in range(1):
        print(test_data[i])
