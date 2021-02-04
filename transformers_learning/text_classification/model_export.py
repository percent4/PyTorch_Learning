# -*- coding: utf-8 -*-
# @Time : 2021/2/4 13:12
# @Author : Jclian91
# @File : model_export.py
# @Place : Yangpu, Shanghai
import json
import torch
import numpy as np
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from params import *
from model_train import convert_text_to_ids, seq_padding

device = torch.device("cpu")
config = BertConfig.from_pretrained("../bert-base-chinese", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
model = BertForSequenceClassification.from_pretrained("../bert-base-chinese", config=config)
model.to(device)
state_dict = torch.load('{}_cls.pth'.format(dataset))
model.load_state_dict(state_dict)

text = "F_A-18超级大黄蜂燃烧器黄昏！"

tokenizer = BertTokenizer("../bert-base-chinese/vocab.txt")
input_ids, token_type_ids = convert_text_to_ids(tokenizer, [text], max_sequence_length)
input_ids = seq_padding(tokenizer, input_ids)
token_type_ids = seq_padding(tokenizer, token_type_ids)
input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
print(input_ids, token_type_ids)
print(input_ids.size())
output = model(input_ids=input_ids, token_type_ids=token_type_ids)


# Export the model
torch.onnx.export(model,  # model being run
                  (input_ids, token_type_ids),  # model input (or a tuple for multiple inputs)
                  "{}_cls.onnx".format(dataset),  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=["input_ids", "token_type_ids"],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input_ids': {0: 'batch_size'},  # variable lenght axes
                                'token_type_ids': {1: 'batch_size'},
                                'output': {0: 'batch_size'}
                                }
                  )