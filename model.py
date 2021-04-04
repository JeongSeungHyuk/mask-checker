import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import models

from efficientnet_pytorch import EfficientNet


class MaskChecker(nn.Module):
    def __init__(self, model):
        super().__init__()

        if model == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Linear(512, 18)
        elif model == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            self.backbone.fc = nn.Linear(512, 18)
        elif model == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Linear(512, 18)
        elif model == 'effnetb0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
            self.backbone._fc = nn.Linear(1280, 18)
        elif model == 'effnetb3':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
            self.backbone._fc = nn.Linear(1536, 18)
        elif model == 'effnetb4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
            self.backbone._fc = nn.Linear(1792, 18)


    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == '__main__':
    model = MaskChecker('resnet34')
    print(model)
