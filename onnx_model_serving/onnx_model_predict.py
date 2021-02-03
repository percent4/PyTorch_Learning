# -*- coding: utf-8 -*-
# @Time : 2021/2/3 20:09
# @Author : Jclian91
# @File : onnx_model_predict.py
# @Place : Yangpu, Shanghai
import onnxruntime
import torch
import numpy as np


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


ort_session = onnxruntime.InferenceSession("iris.onnx")
# compute ONNX Runtime output prediction
x = torch.Tensor([[6.4, 2.8, 5.6, 2.1]])
print("input size: ", to_numpy(x).shape)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
print(ort_outs[0])
print("Exported model has been tested with ONNXRuntime, and the result looks good!")