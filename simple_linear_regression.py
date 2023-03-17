# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


# example 1: simple linear regression, fit a line to the data: Y=w*x+b
# step 1: dataset
x = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
y = np.array([3.1, 5.0, 6.9, 9.2, 13.1])
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9)
inputs = Variable(torch.from_numpy(x.astype("float32")))
outputs = Variable(torch.from_numpy(y.astype("float32")))

for epoch in range(200):
    y_pred = model(inputs)  # forward pass
    loss = cost(y_pred, outputs)    # compute loss
    optimizer.zero_grad()   # backward pass
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print("epoch: {}, loss: {}".format(epoch+1,
                                           loss.data))

# step 4: display model and confirm
plt.figure(figsize=(4, 4))
plt.title("model and dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, y, "ro", label="dataset",
         marker="x", markersize=4)
plt.plot(x,
         model.model.weight.item()*x+model.model.bias.item(),
         label="regression model")
plt.legend()
plt.savefig("simple_linear_regression.png")
