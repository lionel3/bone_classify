import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import gc

from torch.nn import DataParallel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_transforms = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# train_val_dir = "/home/lionel/cuhk/bone_jpg/20171105_img_min_224_flip_negative_1424_89_positive_906_57"

# train_dir = "/home/lionel/cuhk/bone_jpg/20171031_img_min_224_raw_negative_712_89_positive_453_57/train"
train_dir = "/home/lionel/cuhk/bone_jpg/20171105_img_min_224_flip_negative_1424_89_positive_906_57/train"
# train_dir = "/home/lionel/cuhk/bone_jpg/20171106_img_train_both_224_val_min_224/train"
# val_dir = "/home/lionel/cuhk/bone_jpg/20171031_img_min_224_raw_negative_712_89_positive_453_57/val"
val_dir = "/home/lionel/cuhk/bone_jpg/20171105_img_min_224_flip_negative_1424_89_positive_906_57/val"
# val_dir = "/home/lionel/cuhk/bone_jpg/20171106_img_train_both_224_val_min_224/val"

train_dataset = datasets.ImageFolder(train_dir, train_transforms)
val_dataset = datasets.ImageFolder(val_dir, val_transforms)

num_train = len(train_dataset)
num_val = len(val_dataset)
valid_portion = 0.111
num_train_val = num_train + num_val
train_idx = list(range(num_train))
val_idx = list(range(num_val))

# np.random.seed(0)
# np.random.shuffle(train_idx)

# train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
# val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=False
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=False
)

class_names = train_dataset.classes
# print(type(train_dataset))
use_gpu = torch.cuda.is_available()

class my_resnet(torch.nn.Module):
    def __init__(self):
        super(my_resnet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)

        self.task_2 = nn.Sequential()

        self.task_2.add_module("fc1", nn.Lstm())

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        return self.task_1.forward(x)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    epoch_start_time = time.time()

    best_model_wts = model.state_dict()
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train

        # model.train(True)
        model.train()
        scheduler.step()
        train_loss = 0.0
        train_corrects = 0

        train_start_time = time.time()

        for data in train_loader:
            inputs, labels = data
            # print(type(inputs))
            # print(inputs.size())
            # print(type(labels))
            # print(labels.size())
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs.data, 1)
            # print(type(preds))
            # print(type(outputs))
            # print(outputs.size())
            # print(type(labels))
            # print(labels.size())
            loss = criterion(outputs, labels)
            # print(type(loss))
            # print(loss.size())
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
        # model.train(False)
        model.eval()
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

            # print(labels.size())
            # print(inputs.size())

            outputs = model.forward(inputs)
            _, preds = torch.max(outputs.data, 1)
            # print(outputs.size())
            # print(labels.size())
            loss = criterion(outputs, labels)
            val_loss += loss.data[0]
            val_corrects += torch.sum(preds == labels.data)

        val_elapsed_time = time.time() - val_start_time
        print('valid completed in: {:.0f}m{:.0f}s'.format(val_elapsed_time // 60, val_elapsed_time % 60))

        val_average_loss = val_loss / num_val
        val_accuracy = val_corrects / num_val

        print('valid loss: {:.4f} accuracy: {:.4f}'.format(
            val_average_loss, val_accuracy))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            correspond_train_acc = train_accuracy
            best_model_wts = model.state_dict()
        if  val_accuracy == best_val_accuracy:
            if train_accuracy > correspond_train_acc:
                correspond_train_acc = train_accuracy
                best_model_wts = model.state_dict()
        print()

    epoch_elapsed_time = time.time() - epoch_start_time
    print('training completed in {:.0f}m {:.0f}s'.format(epoch_elapsed_time // 60, epoch_elapsed_time % 60))

    print('best val accuracy: {:.4f}'.format(best_val_accuracy))
    print('correspond train acc: {:.4f}'.format(correspond_train_acc))

    model.load_state_dict(best_model_wts)
    # torch.save(model, '20171106_img_min_224_raw_negative_712_89_positive_453_57_resnet_34_batch_32_sgd_1e3_step_5_epoch_25_train_.pth')
    # torch.save(model, '20171106_img_min_224_flip_negative_1424_89_positive_906_57_resnet_50_batch_32_sgd_1e3_step_5_epoch_25_train_.pth')

    # torch.save(model, 'not_pre_20171109_img_train_both_224_val_min_224_crop_flip_negative_74224_89_positive_47920_57_resnet_50_batch_256_sgd_1e3_step_5_epoch_25_train_.pth')
    torch.save(model, '20171115_ada.pth')
    return model


model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
# model_ft = my_resnet()

print(my_resnet)

if use_gpu:
    model_ft = model_ft.cuda()

# model_ft = DataParallel(model_ft)

criterion = nn.CrossEntropyLoss()
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# print(model_ft)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

print('Done!')
print('OK!')
