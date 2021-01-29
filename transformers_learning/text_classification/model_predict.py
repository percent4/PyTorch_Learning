# -*- coding: utf-8 -*-
# @Time : 2021/1/29 13:37
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
import json
import torch
import numpy as np
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from params import *
from model_train import convert_text_to_ids, seq_padding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = BertConfig.from_pretrained("../bert-base-chinese", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
model = BertForSequenceClassification.from_pretrained("../bert-base-chinese", config=config)
model.to(device)
state_dict = torch.load('{}_cls.pth'.format(dataset))
model.load_state_dict(state_dict)

text = "F_A-18超级大黄蜂燃烧器黄昏！"
text = "曾经在车迷圈子中盛传一句话，“传统品牌要是发力新能源，把特斯拉打得渣儿都不剩”。现在实际情况是，不仅仅是传统汽车品牌的纯电动车被打得节节败退，甚至Model 3(参数|图片)还强力侵蚀着本属于传统豪华品牌的汽油车型市场。而就在前天，华晨宝马官方宣布旗下纯电动中型SUV-iX3官方降价7万元，在此之前订购的用户也可以联系经销商退还差价。那么和特斯拉对比显得更有诚意的做法，能不能打败订单已经爆表的Model Y呢？"

tokenizer = BertTokenizer("../bert-base-chinese/vocab.txt")
input_ids, token_type_ids = convert_text_to_ids(tokenizer, [text], max_sequence_length)
# print(input_ids, token_type_ids)
input_ids = seq_padding(tokenizer, input_ids)
token_type_ids = seq_padding(tokenizer, token_type_ids)
input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
output = model(input_ids=input_ids, token_type_ids=token_type_ids)
label_id = np.argmax(output[0].detach().cpu().numpy(), axis=1)[0]

with open("{}_label2id.json".format(dataset), "r", encoding="utf-8") as g:
    label_id_dict = json.loads(g.read())
    id_label_dict = {v: k for k, v in label_id_dict.items()}

print(f"text: {text}")
print(f"predict label: {id_label_dict[label_id]}")
