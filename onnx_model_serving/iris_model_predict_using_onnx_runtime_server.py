# -*- coding: utf-8 -*-
# @Time : 2021/2/3 17:32
# @Author : Jclian91
# @File : iris_model_predict_using_onnx_runtime_server.py
# @Place : Yangpu, Shanghai

import numpy as np
import assets.onnx_ml_pb2 as onnx_ml_pb2
import assets.predict_pb2 as predict_pb2
import requests
from sklearn.datasets import load_iris

# Create request message to be sent to the ORT server
input_tensor = onnx_ml_pb2.TensorProto()
input_tensor.dims.extend([1, 4])
input_tensor.data_type = 1
# input_tensor.raw_data = np.array([[6.4, 2.8, 5.6, 2.1]], dtype=np.float32).tobytes()
input_tensor.raw_data = np.array([[5.7, 3.8, 1.7, 0.3]], dtype=np.float32).tobytes()
print(input_tensor)

request_message = predict_pb2.PredictRequest()

# Use Netron to find out the input name.
request_message.inputs["input"].data_type = input_tensor.data_type
request_message.inputs["input"].dims.extend(input_tensor.dims)
request_message.inputs["input"].raw_data = input_tensor.raw_data

content_type_headers = ['application/x-protobuf', 'application/octet-stream', 'application/vnd.google.protobuf']

for h in content_type_headers:
    request_headers = {
        'Content-Type': h,
        'Accept': 'application/x-protobuf'
    }

# Inference run using ORT server
# Change the number 9001 to the appropriate port number if you had changed it during ORT Server docker instantiation
PORT_NUMBER = 9001  # Change appropriately if needed based on any changes when invoking the server in the pre-requisites
inference_url = "http://192.168.4.193:" + str(PORT_NUMBER) + "/v1/models/default/versions/1:predict"
response = requests.post(inference_url, headers=request_headers, data=request_message.SerializeToString())

# Parse response message
print(response.content)
response_message = predict_pb2.PredictResponse()
response_message.ParseFromString(response.content)

# Use Netron to find out the outputs names.
bboxes = np.frombuffer(response_message.outputs['output'].raw_data, dtype=np.float32)

print('Ouput Dim: ', response_message.outputs['output'].dims)
output_data = np.frombuffer(response_message.outputs['output'].raw_data, dtype=np.float32)
print('Output data: ', output_data)
print("Predict Label: ", load_iris()['target_names'][np.argmax(output_data, axis=0)])
