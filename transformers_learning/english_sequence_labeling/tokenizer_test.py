# -*- coding: utf-8 -*-
# @Time : 2021/1/31 15:06
# @Author : Jclian91
# @File : tokenizer_test.py
# @Place : Yangpu, Shanghai
from transformers import BertForTokenClassification, BertTokenizer

from util import BERT_MODEL_DIR

dict_path = './{}/vocab.txt'.format(BERT_MODEL_DIR)
token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
id_token_dict = {v: k for k, v in token_dict.items()}


# 获取BERT解析后的单词结果
def bert_encode(word):
    code = tokenizer.encode(word)
    print(code)
    bert_test = [id_token_dict[_] for _ in code]
    print(bert_test)
    words = []
    for word in bert_test:
        if word not in ["[CLS]", "[SEP]"]:
            words.append(word)
    return words


words = bert_encode("cosl")
print(words)