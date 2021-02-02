# -*- coding: utf-8 -*-
# @Time : 2021/2/1 14:00
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
import torch
import numpy as np
from torch.utils.data import DataLoader


from params import dataset, BASE_MODEL_DIR, MAX_SEQ_LENGTH, VALID_BATCH_SIZE, ModelTokenizer
from model_train import BERTClass, CustomDataset, idx2tag

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BERTClass()
model.to(dev)
model.load_state_dict(torch.load("{}_{}_ner.pth".format(BASE_MODEL_DIR, dataset)))
tokenizer = ModelTokenizer.from_pretrained('../{}'.format(BASE_MODEL_DIR))


def get_text_predict(text):
    test_set = CustomDataset(tokenizer, [text], [[]], MAX_SEQ_LENGTH)
    test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_loader = DataLoader(test_set, **test_params)

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)
            # print("ids", ids)
            # print("mask", mask)
            # print("target", targets)

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            # print("logits: ", logits)
            tags = [idx2tag[_] for _ in [list(p) for p in np.argmax(logits, axis=2)][0]][1:]

    return tags


if __name__ == '__main__':
    test_text = "11日凌晨，临海市大洋社区工作人员胡大勇和大家一起仍在忙着安置转移的受灾居民。"
    print(get_text_predict(test_text))