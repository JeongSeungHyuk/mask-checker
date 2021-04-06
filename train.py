import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import KFold
from tqdm import tqdm

from dataset import MaskDataset
from model import MaskChecker


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
        y_mask = y[0].to(DEVICE)
        y_gender = y[1].to(DEVICE)
        y_age = y[2].to(DEVICE)

        out_mask, out_gender, out_age = model(x)
        loss_mask = criterion(out_mask, y_mask)
        loss_gender = criterion(out_gender, y_gender)
        loss_age = criterion(out_age, y_age)

        optimizer.zero_grad()
        loss_mask.backward()
        loss_gender.backward()
        loss_age.backward()
        optimizer.step()


def evaluate(model, loader, return_loss=True):
    model.eval()

    correct = 0
    correct_mask = 0
    correct_gender = 0
    correct_age = 0
    total = 0
    sum_loss = 0
    for x, y in tqdm(loader):
        x = x.to(DEVICE)
        y_mask = y[0].to(DEVICE)
        y_gender = y[1].to(DEVICE)
        y_age = y[2].to(DEVICE)

        out_mask, out_gender, out_age = model(x)
        _, y_mask_pred = torch.max(out_mask, 1)
        _, y_gender_pred = torch.max(out_gender, 1)
        _, y_age_pred = torch.max(out_age, 1)

        for i in range(len(y_mask)):
            if y_mask_pred[i] == y_mask[i] and y_gender_pred[i] == y_gender[i] and y_age_pred[i] == y_age[i]:
                correct += 1
            if y_mask_pred[i] == y_mask[i]:
                correct_mask += 1
            if y_gender_pred[i] == y_gender[i]:
                correct_gender += 1
            if y_age_pred[i] == y_age[i]:
                correct_age += 1
        total += len(x)

        if return_loss:
            loss = criterion(out_mask, y_mask) + criterion(out_gender, y_gender) + criterion(out_age, y_age)
            sum_loss += loss.item()

    acc = correct / total
    if return_loss:
        return acc, sum_loss
    else:
        return acc


if __name__ == '__main__':
    set_random_seed(170516)

    model = MaskChecker(MODEL).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
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
        train_acc, train_loss = (0, 0)#evaluate(model, train_loader)
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
