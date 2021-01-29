# -*- coding: utf-8 -*-
# @Time : 2021/1/28 21:45
# @Author : Jclian91
# @File : params.py
# @Place : Yangpu, Shanghai
# 数据集
dataset = "sougou_mini"
train_file = "./data/{}/train.csv".format(dataset)
test_file = "./data/{}/test.csv".format(dataset)

# 模型超参数
max_sequence_length = 256
hidden_dropout_prob = 0.3
num_labels = 5
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 10
batch_size = 32
