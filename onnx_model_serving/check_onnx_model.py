# -*- coding: utf-8 -*-
# @Time : 2021/2/3 19:54
# @Author : Jclian91
# @File : check_onnx_model.py
# @Place : Yangpu, Shanghai
import onnx

# Load the ONNX model
model = onnx.load("iris.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))