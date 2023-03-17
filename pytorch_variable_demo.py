# -*- coding: utf-8 -*-
import torch as T
from torch.autograd import Variable

# Variable会放入一个计算图，然后进行前向传播，反向传播以及自动求导
# 一个Variable里面包含着三个属性，data，grad和creator，
# 其中creator表示得到这个Variable的操作，比如乘法或者加法等等，
# grad表示方向传播的梯度，data表示取出这个Variable里面的数据

# requires_grad 表示是否对其求梯度，默认是True
x = Variable(T.Tensor([3]), requires_grad=True)
y = Variable(T.Tensor([5]), requires_grad=True)
print(x, x.data, x.grad)
print(y, y.data, y.grad)

z = x * x * x + 3 * y + 4

# 对x和y分别求导
print(z.backward())

# x的导数和y的导数
print('dz/dx:{}'.format(x.grad.data))
print('dz/dy:{}'.format(y.grad.data))

# 线性回归示例
# 数据
x_data = [1.0, 2.0, 3.0, 4.0, 6.0]
y_data = [3.1, 5.0, 6.9, 9.2, 13.1]

# 绘制散点图
import matplotlib.pyplot as plt
# plt.scatter(x_data, y_data)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

w = Variable(T.Tensor([0.0]), requires_grad=True)
b = Variable(T.Tensor([0.0]), requires_grad=True)


# forward computation
def forward(x):
    return x * w + b


# define the loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# before training
print("predict(before training)", 5, forward(5).data[0])


eta = 0.01
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        print(l)
        l.backward()
        print("\tgrad:", x_val, y_val, w.grad.data)
        w.data = w.data - eta * w.grad.data
        b.data = b.data - eta * b.grad.data

        # manually set the gradient to zero after updating weight, bias
        w.grad.data.zero_()
        b.grad.data.zero_()

    print("progress: ", epoch, l.data[0])


print(w.data, b.data)
print("predict(atfer training)", 5, forward(5).data[0])
# 绘制拟合的直线
import numpy as np
plt.scatter(x_data, y_data)
x = np.linspace(1, 6, 100)
y = w.data.numpy()[0] * x + b.data.numpy()[0]
plt.plot(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
