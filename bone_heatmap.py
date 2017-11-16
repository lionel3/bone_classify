from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2

import os

if torch.cuda.is_available():
    use_gpu = True

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def info(variable):
    print(type(variable))
    if isinstance(variable, np.ndarray):
        print(variable.shape)
    elif isinstance(variable, list):
        print(len(variable))
    elif isinstance(variable, tuple):
        print(len(variable))
    elif isinstance(variable, dict):
        print(len(dict))
    elif isinstance(variable, torch.FloatTensor):
        print(variable.shape)
    elif isinstance(variable, torch.cuda.FloatTensor):
        print(variable.shape)


def get_all_files(root_dir, file_paths, file_names):
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
        else:
            get_all_files(path, file_paths, file_names)


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
        self.task_1 = nn.Sequential()

        self.task_1.add_module("fc1", nn.Linear(2048, 100))
        self.task_1.add_module("fc2", nn.Linear(100, 2))

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        return self.task_1.forward(x)


mode = 'negative'
root_dir = "/home/lionel/cuhk/bone_jpg/20171031_img_min_224_raw_negative_712_89_positive_453_57/val/" + mode
val_file_paths = []
val_file_names = []
get_all_files(root_dir, val_file_paths, val_file_names)

val_file_names.sort()
val_file_paths.sort()

print('file_names_size:', len(val_file_names))
print('file_paths_size:', len(val_file_paths))

# for i in range(0, len(val_file_names)):
# print(val_file_paths[i], val_file_names[i])


finalconv_name = 'layer4'


# net = torch.load('/home/lionel/pytorch/image_classification/20171106_img_train_both_224_'
#                  'val_min_224_crop_flip_negative_74224_89_positive_47920_57_resnet_50_'
#                  'batch_256_sgd_1e3_step_5_epoch_25_train_1_val_8904.pth',
#                  map_location=lambda storage, loc: storage)
net = torch.load('/home/lionel/pytorch/bone_classify/20171106_img_train_both_224_'
                 'val_min_224_crop_flip_negative_74224_89_positive_47920_57_resnet_50_'
                 'batch_256_sgd_1e3_step_5_epoch_25_train_1_val_8904.pth')
# net = torch.load('/home/lionel/pytorch/image_classification/20171106_img_min_224_flip_'
#                  'negative_1424_89_positive_906_57_resnet_50_batch_32_sgd_1e3_step_5_'
#                  'epoch_25_train_9695_val_8904.pth')
# print(net.module.share._modules.get(finalconv_name))
# print(net.share)
# net = torch.load('/home/lionel/pytorch/image_classification/20171027_bone_ct_resnet_152_25.pth', map_location=lambda storage, loc: storage)
# print(net._modules('my_resnet'))

net = net.module
print(net)

net.eval()

# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net.share._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze((params[-2].cpu().data.numpy()).dot(params[-4].cpu().data.numpy()))
print(weight_softmax.shape)
class_names = ['negative', 'positive']

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

preprocess = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

scoremap_dir = '/home/lionel/cuhk/bone_jpg/scoremap/' + mode
heatmap_dir = '/home/lionel/cuhk/bone_jpg/heatmap/' + mode

record_txt = '/home/lionel/cuhk/bone_jpg/' + mode + '_record.txt'

if os.path.exists(scoremap_dir):
    pass
else:
    os.makedirs(scoremap_dir)

if os.path.exists(heatmap_dir):
    pass
else:
    os.makedirs(heatmap_dir)

classes = {0: 'negative', 1: 'positive'}

f = open(record_txt, 'w')
f.write(mode)
f.write('\n')

for i in range(0, len(val_file_names)):
    img_pil = Image.open(val_file_paths[i])
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    logit = net(img_variable.cuda())

    h_x = F.softmax(logit).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # for i in range(0, 2):
    #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    # print(type(features_blobs))
    # print(len(features_blobs))
    # print(features_blobs)
    # print(type(weight_softmax))
    # print(weight_softmax.shape)


    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # print('output result.jpg for the top1 prediction: %s' % classes[idx[0]])

    img = cv2.imread(val_file_paths[i])
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)

    scoremap = heatmap * 0.3 + img * 0.5

    scoremap_name = scoremap_dir + '/' + val_file_names[i]

    heatmap_name = heatmap_dir + '/' + val_file_names[i]

    cv2.imwrite(scoremap_name, scoremap)
    cv2.imwrite(heatmap_name, heatmap)

    f.write(val_file_names[i])
    f.write('    ')
    f.write(str(idx[0]))
    f.write('    ')
    f.write(str(probs[0]))
    f.write('\n')

f.close()

print('Done!')
