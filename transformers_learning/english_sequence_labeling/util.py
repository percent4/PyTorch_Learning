# -*- coding: utf-8 -*-
# @Time : 2020/12/24 13:26
# @Author : Jclian91
# @File : util.py
# @Place : Yangpu, Shanghai

# 数据相关的配置
event_type = "conll2003"

train_file_path = "./data/%s.train" % event_type
test_file_path = "./data/%s.test" % event_type

# 模型相关的配置
BERT_MODEL_DIR = "./bert-base-uncased"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-05
