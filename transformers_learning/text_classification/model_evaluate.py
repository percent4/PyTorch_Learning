# -*- coding: utf-8 -*-
# @Time : 2021/1/29 14:13
# @Author : Jclian91
# @File : model_evaluate.py
# @Place : Yangpu, Shanghai
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from params import *
from model_train import convert_text_to_ids, seq_padding, test_file

# read label id dict
with open("{}_label2id.json".format(dataset), "r", encoding="utf-8") as g:
    label_id_dict = json.loads(g.read())
    id_label_dict = {v: k for k, v in label_id_dict.items()}

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = BertConfig.from_pretrained("../bert-base-chinese", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
model = BertForSequenceClassification.from_pretrained("../bert-base-chinese", config=config)
model.to(device)
state_dict = torch.load('{}_cls.pth'.format(dataset))
model.load_state_dict(state_dict)
tokenizer = BertTokenizer("../bert-base-chinese/vocab.txt")

# read test file
test_df = pd.read_csv(test_file)
contents, true_labels = test_df["content"].tolist(), test_df["label"].tolist()

# model evaluate
pred_labels = []
for i, text in enumerate(contents):
    print("predict {} samples".format(i+1))
    input_ids, token_type_ids = convert_text_to_ids(tokenizer, [text], max_sequence_length)
    # print(input_ids, token_type_ids)
    input_ids = seq_padding(tokenizer, input_ids)
    token_type_ids = seq_padding(tokenizer, token_type_ids)
    input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
    input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
    output = model(input_ids=input_ids, token_type_ids=token_type_ids)
    label_id = np.argmax(output[0].detach().cpu().numpy(), axis=1)[0]
    pred_labels.append(id_label_dict[label_id])


# print evaluate output
print(classification_report(true_labels, pred_labels, digits=4))