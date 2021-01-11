# -*- coding: utf-8 -*-
# @Time : 2021/1/11 13:50
# @Author : Jclian91
# @File : cv_pretrained_models.py
# @Place : Yangpu, Shanghai
# use pretrained CV models to predict on a new image
import torch
from PIL import Image
from torchvision import models
from torchvision import transforms

print(dir(models))

# alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)

# data preprocess
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])

# predic on a new image
img = Image.open("dog.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)
# print(out)

# show the result
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
_, index = torch.max(out, 1)
print(labels[index.numpy()[0]])