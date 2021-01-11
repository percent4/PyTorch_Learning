# -*- coding: utf-8 -*-
# @Time : 2021/1/7 18:22
# @Author : Jclian91
# @File : pytorch_tensor_demo.py
# @Place : Yangpu, Shanghai
import torch as T
import numpy as np

# 检测是否为PyTorch中的向量（Tensor）
x = [12, 23, 34, 45, 56, 67, 78, 89]
# checks whether the object is a tensor object
print(T.is_tensor(x))
# checks whether the object is stored as tensor object
print(T.is_storage(x))

y = T.randn(2, 2, 3)
print(y)
print(T.is_tensor(y))
print(T.is_storage(y))
# size of tensor
print(y.size())
# the total number of elements in the input tensor
print(T.numel(y))

# zeros函数
z = T.zeros(4, 5)
print(z)
print(z.size())
print(T.numel(z))

# eye函数
w1 = T.eye(3, 4)
print(w1)
print(w1.size())
print(T.numel(w1))
w2 = T.eye(5, 4)
print(w2)
print(w2.size())
print(T.numel(w2))

# Torch can read from a NumPy array format, not list
x1 = np.array(x)
t = T.from_numpy(x1)
print(t)
print(T.is_tensor(t))
print(T.is_storage(t))

# Torch Tensor convert to numpy.ndarrays
print("*"*50)
print(t.data)
print(type(t))
print(type(t.data))
x2 = t.numpy()
print(x2)
print(type(x2))

# 利用Tensor函数构建tensor
tensor1 = T.Tensor([[1, 2, 3], [4, 5, 6]])
print(tensor1)
print(T.is_tensor(tensor1))
print(T.is_storage(tensor1))

# find maximum and minimum of tensor
d = T.randn(4, 5)
print(d)
# dim: 0 means along the column, 1 row
print(T.argmin(d, dim=0))
print(T.argmax(d, dim=1))

# concatenate two tensors
t1 = T.randn(2, 4)
t2 = T.randn(3, 4)
print(T.cat((t1, t2)))

# transpose
e = T.randn(2, 3)
print(e)
print(e.t())
print(e.transpose(1, 0))

# 矩阵的点乘, x.mul(y) 或者 x * y
h = T.FloatTensor([[1, 2], [3, 4], [5, 6]])
print(h)
print(h.mul(h))
print(h * h)

print(h.mm(h.t()))

# 向量的CPU和GPU使用
x = T.FloatTensor([[1, 2, 3], [4, 5, 6]])
y = np.matrix([[2, 2, 2], [2, 2, 2]], dtype="float32")
if T.cuda.is_available():
    x = x.cuda()
    y = T.from_numpy(y).cuda()
    z = x+y
    print(z)
    print(z.cpu())
else:
    print("no cuda available")

"""
output:
tensor([[3., 4., 5.],
        [6., 7., 8.]], device='cuda:0')
tensor([[3., 4., 5.],
        [6., 7., 8.]])
"""


