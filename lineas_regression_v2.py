# -*- coding: utf-8 -*-
# @Time : 2021/1/11 11:00
# @Author : Jclian91
# @File : lineas_regression_v2.py
# @Place : Yangpu, Shanghai
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


# example 1: simple linear regression, fit a line to the data: Y=w*x+b
np.random.seed(2021)
torch.manual_seed(2021)

# step 1: dataset
w, b = 2, 3
x = np.linspace(0, 10, 100)
y = w*x+b+np.random.rand(100)*2
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


# step 2: model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegressionModel, self).__init__()
        self.model = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred


model = LinearRegressionModel(in_dim=1, out_dim=1)

# step 3: training
cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
inputs = Variable(torch.from_numpy(x.astype("float32")))
outputs = Variable(torch.from_numpy(y.astype("float32")))

for epoch in range(200):
    # forward pass
    y_pred = model(inputs)
    # compute loss
    loss = cost(y_pred, outputs)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print("epoch: {}, loss: {}".format(epoch+1, loss.data))

# step 4: display model and confirm
plt.figure(figsize=(4, 4))
plt.title("model and dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, y, "ro", label="dataset", marker="x", markersize=4)
plt.plot(x, model.model.weight.item()*x+model.model.bias.item(), label="regression model")
plt.legend()
plt.savefig("linear_regression.png")

