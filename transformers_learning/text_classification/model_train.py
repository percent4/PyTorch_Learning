# -*- coding: utf-8 -*-
# @Time : 2021/1/27 18:48
# @Author : Jclian91
# @File : model_train.py
# @Place : Yangpu, Shanghai
import json
import torch
import torch.nn as nn
from transformers import AdamW
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from params import *


class ClsDataset(Dataset):
    def __init__(self, path_to_file, is_training=True):
        self.dataset = pd.read_csv(path_to_file)
        self.is_training = is_training
        if self.is_training:
            self.labels = list(self.dataset["label"].unique())
            with open("{}_label2id.json".format(dataset), "w", encoding="utf-8") as f:
                f.write(json.dumps(dict(zip(self.labels, range(len(self.labels)))), ensure_ascii=False, indent=2))
        else:
            with open("{}_label2id.json".format(dataset), "r", encoding="utf-8") as g:
                self.label_id_dict = json.loads(g.read())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "content"]
        label = self.dataset.loc[idx, "label"]
        if self.is_training:
            label_id = self.labels.index(label)
        else:
            label_id = self.label_id_dict[label]
        sample = {"content": text, "label_id": label_id}
        return sample


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
    if len(X) <= 1:
        return torch.tensor(X)
    X = torch.Tensor([x + [pad_id] * (max_sequence_length - len(x)) if len(x) < max_sequence_length else x for x in X])
    return X


def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(iterator):
        label = batch["label_id"]
        text = batch["content"]
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, text, max_sequence_length)
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        # 标签形状为 (batch_size, 1)
        label = label.unsqueeze(1)
        # 需要 LongTensor
        input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
        # 梯度清零
        optimizer.zero_grad()
        # 迁移到GPU
        input_ids, token_type_ids, label = input_ids.to(device), token_type_ids.to(device), label.to(device)
        output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
        y_pred_prob = output[1]
        y_pred_label = y_pred_prob.argmax(dim=1)
        # 计算loss
        # 这个 loss 和 output[0] 是一样的
        loss = criterion(y_pred_prob.view(-1, num_labels), label.view(-1))
        print("bacth {}, loss: {}".format(i, loss.data))
        # 计算acc
        acc = ((y_pred_label == label.view(-1)).sum()).item()
        # 反向传播
        loss.backward()
        optimizer.step()
        # epoch 中的 loss 和 acc 累加
        epoch_loss += loss.item()
        epoch_acc += acc
        if i % 20 == 0:
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(label)))
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)


def evaluate(model, iterator,  criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            label = batch["label_id"]
            text = batch["content"]
            input_ids, token_type_ids = convert_text_to_ids(tokenizer, text, max_sequence_length)
            input_ids = seq_padding(tokenizer, input_ids)
            token_type_ids = seq_padding(tokenizer, token_type_ids)
            label = label.unsqueeze(1)
            input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
            input_ids, token_type_ids, label = input_ids.to(device), token_type_ids.to(device), label.to(device)
            output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
            y_pred_label = output[1].argmax(dim=1)
            loss = output[0]
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            epoch_loss += loss.item()
            epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)


if __name__ == '__main__':
    # 使用GPU
    # 通过model.to(device)的方式使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = BertConfig.from_pretrained("../bert-base-chinese", num_labels=num_labels,
                                        hidden_dropout_prob=hidden_dropout_prob)
    model = BertForSequenceClassification.from_pretrained("../bert-base-chinese", config=config)
    model.to(device)

    # 加载数据集
    sentiment_train_set = ClsDataset(train_file)
    sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=True)

    sentiment_valid_set = ClsDataset(test_file, is_training=False)
    sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False)

    tokenizer = BertTokenizer("../bert-base-chinese/vocab.txt")

    # 定义优化器和损失函数
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("start training...")
    for i in range(epochs):
        train_loss, train_acc = train(model, sentiment_train_loader, optimizer, criterion, device)
        print("train loss: ", train_loss, "\t", "train acc:", train_acc)
        valid_loss, valid_acc = evaluate(model, sentiment_valid_loader, criterion, device)
        print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
        torch.save(model.state_dict(), '{}_cls.pth'.format(dataset))
    print("end training...")