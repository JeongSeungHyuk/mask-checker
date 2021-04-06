import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import models

from efficientnet_pytorch import EfficientNet


class MaskChecker(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.mask = models.resnet18(pretrained=True)
        self.mask.fc = nn.Linear(512, 3)

        self.gender = models.resnet18(pretrained=True)
        self.gender.fc = nn.Linear(512, 2)

        self.age = models.resnet18(pretrained=True)
        self.age.fc = nn.Linear(512, 3)


    def forward(self, x):
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age


if __name__ == '__main__':
    model = MaskChecker('resnet34')
    print(model)
