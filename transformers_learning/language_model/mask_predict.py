# -*- coding: utf-8 -*-
# @Time : 2021/1/27 17:10
# @Author : Jclian91
# @File : mask_predict.py
# @Place : Yangpu, Shanghai
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("../ernie-1.0")
model = AutoModelWithLMHead.from_pretrained("../ernie-1.0")


sequence = f"地球表面积5.1亿平方公里，其中71%为{tokenizer.mask_token}{tokenizer.mask_token}，29%为陆地，在太空上看地球呈蓝色"
print(sequence)

input = tokenizer.encode(sequence, return_tensors="pt")
# print(input)
mask_token_index = torch.where(input==tokenizer.mask_token_id)[1]
print("mask index: ", mask_token_index.numpy())

token_logits = model(input)[0]
mask_token_logits = token_logits[0, mask_token_index, :]

for i in range(len(mask_token_index.numpy())):
    top_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[i].tolist()
    for token in top_tokens:
        token_input = input.numpy()[0]
        token_input[mask_token_index.numpy()[i]] = token
        print("".join([tokenizer.decode([token]) for token in token_input][1:-1]))