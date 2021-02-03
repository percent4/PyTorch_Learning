# -*- coding: utf-8 -*-
# @Time : 2021/2/3 21:14
# @Author : Jclian91
# @File : captcha_recognition_model_predict_using_onnx_runtime_server.py
# @Place : Yangpu, Shanghai
import numpy as np
import assets.onnx_ml_pb2 as onnx_ml_pb2
import assets.predict_pb2 as predict_pb2
import requests
from PIL import Image
import time


start_time = time.time()
# Load the raw image
input_shape = (1, 3, 50, 22)
img = Image.open("assets/code2.png")


# Preprocess and normalize the image
img_data = np.array(img)
img_data = np.transpose(img_data, [2, 0, 1])
img_data = np.expand_dims(img_data, 0)
norm_img_data = np.zeros(img_data.shape).astype('float32')
for i in range(img_data.shape[1]):
    norm_img_data[:, i, :, :] = img_data[:, i, :, :] / 255

# Create request message to be sent to the ORT server
input_tensor = onnx_ml_pb2.TensorProto()
input_tensor.dims.extend(norm_img_data.shape)
input_tensor.data_type = 1
input_tensor.raw_data = norm_img_data.tobytes()

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
PORT_NUMBER = 9001 # Change appropriately if needed based on any changes when invoking the server in the pre-requisites
inference_url = "http://192.168.4.193:" + str(PORT_NUMBER) + "/v1/models/default/versions/1:predict"
response = requests.post(inference_url, headers=request_headers, data=request_message.SerializeToString())

# Parse response message
response_message = predict_pb2.PredictResponse()
response_message.ParseFromString(response.content)

# Use Netron to find out the outputs names.
output = np.frombuffer(response_message.outputs['output1'].raw_data, dtype=np.float32)
print("first digit: ", np.argmax(output, axis=0))
output = np.frombuffer(response_message.outputs['output2'].raw_data, dtype=np.float32)
print("second digit: ", np.argmax(output, axis=0))
output = np.frombuffer(response_message.outputs['output3'].raw_data, dtype=np.float32)
print("third digit: ", np.argmax(output, axis=0))
output = np.frombuffer(response_message.outputs['output4'].raw_data, dtype=np.float32)
print("fourth digit: ", np.argmax(output, axis=0))
end_time = time.time()
print("cost time: ", (end_time-start_time))
