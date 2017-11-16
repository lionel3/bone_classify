import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os

from torch.nn import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# one_folder train val

train_val_dir = "/home/lionel/cuhk/bone_jpg/20171031_img_min_224"
train_dataset = datasets.ImageFolder(train_val_dir, train_transforms)
val_dataset = datasets.ImageFolder(train_val_dir, val_transforms)

num_train_val = len(train_dataset)

valid_portion = 0.111

indices = list(range(num_train_val))
split = int(np.floor(valid_portion * num_train_val))

num_val = split
num_train = num_train_val - num_val


train_idx, val_idx = indices[split:], indices[:split]

np.random.seed(0)
np.random.shuffle(train_idx)

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=120,
    # shuffle=True,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=False
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=120,
    # shuffle=False,
    sampler=val_sampler,
    num_workers=4,
    pin_memory=False
)

class_names = train_dataset.classes

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    epoch_start_time = time.time()

    best_model_wts = model.state_dict()
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train

        scheduler.step()
        model.train(True)
        train_loss = 0.0
        train_corrects = 0

        train_start_time = time.time()

        for data in train_loader:
            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            train_corrects += torch.sum(preds == labels.data)

        train_elapsed_time = time.time() - train_start_time
        print('train completed in: {:.0f}m{:.0f}s'.format(train_elapsed_time // 60, train_elapsed_time % 60))

        train_average_loss = train_loss / num_train
        train_accuracy = train_corrects / num_train

        print('train loss: {:.4f} accuracy: {:.4f}'.format(
            train_average_loss, train_accuracy))

        # val

        model.train(False)
        val_loss = 0.0
        val_corrects = 0

        val_start_time = time.time()

        for data in val_loader:

            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.data[0]
            val_corrects += torch.sum(preds == labels.data)

        val_elapsed_time = time.time() - val_start_time
        print('val completed in: {:.0f}m{:.0f}s'.format(val_elapsed_time // 60, val_elapsed_time % 60))

        val_average_loss = val_loss / num_val
        val_accuracy = val_corrects / num_val

        print('val loss: {:.4f} accuracy: {:.4f}'.format(
            val_average_loss, val_accuracy))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            correspond_train_acc = train_accuracy
            best_model_wts = model.state_dict()

        print()

    epoch_elapsed_time = time.time() - epoch_start_time
    print('training completed in {:.0f}m {:.0f}s'.format(epoch_elapsed_time // 60, epoch_elapsed_time % 60))

    print('best val accuracy: {:.4f}'.format(best_val_accuracy))
    print('correspond train acc: {:.4f}'.format(correspond_train_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model, '20171102_bone_img_both_224_resnet_50_epoch_25_random_0_batch_80_val_0dot111.pth')
    return model

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
if use_gpu:
    model_ft = model_ft.cuda()

model_ft = DataParallel(model_ft)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# print(model_ft)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)

print('Done!')
print('OK!')
