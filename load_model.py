# -*- coding: utf-8 -*-
# @Time : 2021/1/25 18:45
# @Author : Jclian91
# @File : load_model.py
# @Place : Yangpu, Shanghai
import torch
from torch import Tensor

from save_model import MLP

model = MLP(34)
state_dict = torch.load('./binary_classification.pth')
model.load_state_dict(state_dict)
print(model)
# make a single prediction (expect class=1)
row = [1, 0, 0.99539, -0.05889, 0.85243, 0.02306, 0.83398, -0.37708, 1, 0.03760, 0.85243, -0.17755, 0.59755, -0.44945,
       0.60536, -0.38223, 0.84356, -0.38542, 0.58212, -0.32192, 0.56971, -0.29674, 0.36946, -0.47357, 0.56811, -0.51171,
       0.41078, -0.46168, 0.21266, -0.34090, 0.42267, -0.54487, 0.18641, -0.45300]
row = Tensor([row])
# make prediction
yhat = model(row)
# retrieve numpy array
yhat = yhat.detach().numpy()
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))