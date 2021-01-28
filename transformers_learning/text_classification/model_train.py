# -*- coding: utf-8 -*-
# @Time : 2021/1/27 18:48
# @Author : Jclian91
# @File : model_train.py
# @Place : Yangpu, Shanghai
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# load data
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path, is_train=False ):
        # load the csv file as a dataframe
        df = pd.read_csv(path)
        # store the inputs and outputs
        content, label = df["content"], df["label"]
        # store label2id mapping
        labels = list(label.unique())
        if is_train:
            label_id_dict = dict(zip(labels, range(len(labels))))
            with open("label2id.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(label_id_dict, ensure_ascii=False, indent=2))

        self.X = None
        self.y = label.apply(lambda x: labels.index(x))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# prepare data
CSVDataset("./data/sougou_mini/train.csv", is_train=True)

tokenizer = AutoTokenizer.from_pretrained("../bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained("../bert-base-chinese")
print(model)

# classes = ["not paraphrase", "is paraphrase"]
#
# sequence_0 = "The company HuggingFace is based in New York City"
# sequence_1 = "Apples are especially bad for your health"
# sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
#
# paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, return_tensors="pt")
# print(paraphrase)
# not_paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, return_tensors="pt")
#
# paraphrase_classification_logits = model(**paraphrase)[0]
# not_paraphrase_classification_logits = model(**not_paraphrase)[0]
#
# paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
# not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]
#
# print("Should be paraphrase")
# for i in range(len(classes)):
#     print(f"{classes[i]}: {round(paraphrase_results[i] * 100)}%")
#
# print("\nShould not be paraphrase")
# for i in range(len(classes)):
#     print(f"{classes[i]}: {round(not_paraphrase_results[i] * 100)}%")