# -*- coding: utf-8 -*-
# @Time : 2021/1/29 22:48
# @Author : Jclian91
# @File : token_test.py
# @Place : Yangpu, Shanghai
from transformers import BertTokenizerFast, AutoTokenizer, AutoConfig

from load_data import read_data
from params import train_file_path, test_file_path

train_sents, train_tags = read_data(train_file_path)

tokenizer = AutoTokenizer.from_pretrained("../bert-base-chinese")
for sent, tag in zip(train_sents, train_tags):
    code = tokenizer.encode(sent.lower())
    # print(code)
    tokens = [tokenizer.convert_ids_to_tokens(_) for _ in code]
    if "[UNK]" in tokens:
        print(sent)
        print(tag)
        print(tokens)

# text = "另外意大利的PlayGeneration杂志也刚刚给出了92%的高分。".lower()
# true_tag = []
# i = 0
# for token in code[1:-1]:
#     true_tag.append(tags[i])
#     i += len(tokenizer.convert_ids_to_tokens(token))
# print(true_tag)
