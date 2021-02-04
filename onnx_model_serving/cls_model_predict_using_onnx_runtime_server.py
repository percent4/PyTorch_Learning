# -*- coding: utf-8 -*-
# @Time : 2021/2/3 17:32
# @Author : Jclian91
# @File : iris_model_predict_using_onnx_runtime_server.py
# @Place : Yangpu, Shanghai
import torch
import numpy as np
import assets.onnx_ml_pb2 as onnx_ml_pb2
import assets.predict_pb2 as predict_pb2
import requests
from transformers import BertTokenizer

max_sequence_length = 256


def convert_text_to_ids(tokenizer, text, max_len):
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
    elif isinstance(text, list):
        input_ids = []
        token_type_ids = []
        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    else:
        print("Unexpected input")
    return input_ids, token_type_ids


def seq_padding(tokenizer, X):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    X = torch.Tensor([x + [pad_id] * (max_sequence_length - len(x)) if len(x) < max_sequence_length else x for x in X])
    return X


text = "F_A-18超级大黄蜂燃烧器黄昏！"
text = "最近，南京有多位学生家长反映，他们给孩子在南杨作文培训机构报了辅导班，可带孩子来上课的时候发现，机构竟然大门紧闭，人去楼空。这样的情形让不少家长慌了神。" \
       "1月21号上午，在南京鼓楼区华彩荟四楼的南杨作文培训机构门前，记者见到了多位前来维权的学生家长。陈女士告诉记者，去年10月份，她交了4800多块钱，" \
       "为孩子报了南杨作文的辅导班。"
text = "北京时间2月4日消息，保罗-乔治全场轰下36分6次助攻，科怀-伦纳德贡献24分和6个篮板，带领快船队客场以121-99击败骑士队。四节具体比分为" \
       "（快船在前）：33-25，19-24，36-25，33-25。快船：科怀-伦纳德24分6篮板3助攻，塞尔吉-伊巴卡14分，保罗-乔治36分4篮板6助攻，祖巴茨10分" \
       "16篮板，路易斯-威廉姆斯15分"
text = "2月3日，澎湃新闻从国家企业信用信息公示系统发现，特斯拉超级工厂项目（一期）第二阶段环境影响报告书已获临港地区开发建设管理委员会许可。" \
       "这意味着工厂（一期）第二阶段的动工已被批准，工厂建设及新车型生产启动在即。特斯拉环境影响报告书显示，该项目新增纯电动汽车整车冲压、涂装、" \
       "焊装和总装等生产工艺。项目建成后，将新增Model 3、Model Y车型及相关衍生车型生产能力。"

tokenizer = BertTokenizer("../transformers_learning/bert-base-chinese/vocab.txt")
input_ids, token_type_ids = convert_text_to_ids(tokenizer, [text], max_sequence_length)
input_ids = seq_padding(tokenizer, input_ids)
token_type_ids = seq_padding(tokenizer, token_type_ids)
input_ids, token_type_ids = input_ids.long(), token_type_ids.long()


# Create request message to be sent to the ORT server
input_tensor1 = onnx_ml_pb2.TensorProto()
input_tensor1.dims.extend([1, 256])
input_tensor1.data_type = 7
input_tensor1.raw_data = input_ids.numpy().astype(np.int64).tobytes()

input_tensor2 = onnx_ml_pb2.TensorProto()
input_tensor2.dims.extend([1, 256])
input_tensor2.data_type = 7
input_tensor2.raw_data = token_type_ids.numpy().astype(np.int64).tobytes()

request_message = predict_pb2.PredictRequest()

# Use Netron to find out the input name.
request_message.inputs["input_ids"].data_type = input_tensor1.data_type
request_message.inputs["input_ids"].dims.extend(input_tensor1.dims)
request_message.inputs["input_ids"].raw_data = input_tensor1.raw_data

request_message.inputs["token_type_ids"].data_type = input_tensor2.data_type
request_message.inputs["token_type_ids"].dims.extend(input_tensor2.dims)
request_message.inputs["token_type_ids"].raw_data = input_tensor2.raw_data

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
# print(response.content)
response_message = predict_pb2.PredictResponse()
response_message.ParseFromString(response.content)

# Use Netron to find out the outputs names.
bboxes = np.frombuffer(response_message.outputs['output'].raw_data, dtype=np.float32)

print("input text: ", text)
print('Ouput Dim: ', response_message.outputs['output'].dims)
output_data = np.frombuffer(response_message.outputs['output'].raw_data, dtype=np.float32)
print('Output data: ', output_data)
labels = ["体育", "健康", "军事", "教育", "汽车"]
print("Predict Label: ", labels[np.argmax(output_data, axis=0)])
