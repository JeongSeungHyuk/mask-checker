import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import csv
from PIL import Image

from tqdm import tqdm

from dataset import MaskDataset, MaskTestDataset
from model import MaskChecker
from train import evaluate


DEVICE = torch.device('cuda:0')
NUM_WORKERS = 4

BATCH_SIZE = 16

MODEL_PATH = 'checkpoints/resnet34_lr0.0003_b16_tol5_1.pt'
RESULT_PATH = 'submission.csv'


valid_dataset = MaskDataset(train=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_dataset = MaskTestDataset()
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

model = torch.load(MODEL_PATH).to(DEVICE)
model.eval()


print('validation')
acc = evaluate(model, valid_loader, return_loss=False)
print(f'acc: {acc * 100}')


print('inference')

result = []
result.append(['ImageID', 'ans'])

for x, path in tqdm(test_loader):
    x = x.to(DEVICE)

    out_mask, out_gender, out_age = model(x)
    _, y_mask_pred = torch.max(out_mask, 1)
    _, y_gender_pred = torch.max(out_gender, 1)
    _, y_age_pred = torch.max(out_age, 1)
    label = (y_mask_pred * 6) + (y_gender_pred * 3) + y_age_pred

    for i in range(len(x)):
        result.append([path[i], label[i].item()])

with open(RESULT_PATH, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(result)
