# -*- coding: utf-8 -*-
# @Time : 2021/1/13 17:00
# @Author : Jclian91
# @File : load_different_file_format_into_tensor.py
# @Place : Yangpu, Shanghai
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50)

import imageio

img_arr = imageio.imread('./data/bobby.jpg')
print(img_arr.shape)

img = torch.from_numpy(img_arr)
out = img.permute(2, 0, 1)
print(out)