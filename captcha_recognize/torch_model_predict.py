# -*- coding: utf-8 -*-
# @Time : 2021/1/26 14:53
# @Author : Jclian91
# @File : torch_model_predict.py
# @Place : Yangpu, Shanghai
import cv2
import torch
import numpy as np
from torch import Tensor

from torch_model_train import CNNModel


model = CNNModel()
state_dict = torch.load('captcha_recognize.pth')
model.load_state_dict(state_dict)
# print(model)

# predict on a new image
image_pixel = cv2.imread("new_images/code12.png", 1)/255
test_X = Tensor([torch.from_numpy(image_pixel).permute(2, 0, 1).numpy()])
predicts = model(test_X)
# retrieve numpy array
result = []
for predict in predicts:
    predict = predict.detach().numpy()
    result.append(np.argmax(predict, axis=1)[0])

print("predict: %s" % result)