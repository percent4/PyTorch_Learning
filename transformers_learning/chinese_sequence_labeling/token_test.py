# -*- coding: utf-8 -*-
# @Time : 2021/1/29 22:48
# @Author : Jclian91
# @File : token_test.py
# @Place : Yangpu, Shanghai
from transformers import BertTokenizerFast, AutoTokenizer, AutoConfig


tokenizer = AutoTokenizer.from_pretrained("../chinese-roberta-wwm-ext-large")

text = "另外意大利的PlayGeneration杂志也刚刚给出了92%的高分。".lower()
code = tokenizer.encode(text)
print(code)
print([tokenizer.convert_ids_to_tokens(_) for _ in code])
# true_tag = []
# i = 0
# for token in code[1:-1]:
#     true_tag.append(tags[i])
#     i += len(tokenizer.convert_ids_to_tokens(token))
# print(true_tag)
