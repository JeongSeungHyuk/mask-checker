import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import KFold
from tqdm import tqdm

from dataset import MaskDataset
from model import MaskChecker
from loss import *


DEVICE = torch.device('cuda:0')
NUM_WORKERS = 4

NUM_EPOCHS = 30
TOLERENCE = 5
BATCH_SIZE = 16

MODEL = 'resnet34'
LEARNING_RATE = 3e-4


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, loader):
    model.train()

    for x, y in tqdm(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with amp.autocast():
            output = model(x)
            loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, loader, return_loss=True):
    model.eval()

    correct = 0
    total = 0
    sum_loss = 0
    for x, y in tqdm(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with amp.autocast():
            output = model(x)
            _, y_pred = torch.max(output, 1)

        correct += (y_pred == y).sum().item()
        total += len(x)

        if return_loss:
            loss = criterion(output, y)
            sum_loss += loss.item()

    acc = correct / total
    if return_loss:
        return acc, sum_loss
    else:
        return acc


if __name__ == '__main__':
    set_random_seed(170516)

    model = MaskChecker(MODEL).to(DEVICE)
    criterion = LabelSmoothingLoss(18, 0.2) #nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

    train_dataset = MaskDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    valid_dataset = MaskDataset(train=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    logger = SummaryWriter(log_dir='logs')

    best_valid_acc = 0
    fails = 0
    for epoch in range(NUM_EPOCHS):
        print(f'<Epoch {epoch}>')

        print('training')
        train(model, train_loader)

        print('validation')
        train_acc, train_loss = evaluate(model, train_loader)
        valid_acc, valid_loss = evaluate(model, valid_loader)
        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            fails = 0
            torch.save(model, f'checkpoints/{MODEL}_lr{LEARNING_RATE}_b{BATCH_SIZE}_tol{TOLERENCE}_{epoch}.pt')
        else:
            fails += 1
        print(f'train_acc: {train_acc * 100} | valid_acc: {valid_acc * 100} | best_valid_acc: {best_valid_acc * 100} | fails: {fails}/{TOLERENCE}')

        logger.add_scalar("train loss", train_loss, epoch)
        logger.add_scalar("train acc", train_acc, epoch)
        logger.add_scalar("valid loss", valid_loss, epoch)
        logger.add_scalar("valid acc", valid_acc, epoch)

        if fails >= TOLERENCE:
            break

        scheduler.step(valid_loss)
