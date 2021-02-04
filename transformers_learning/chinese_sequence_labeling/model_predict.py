# -*- coding: utf-8 -*-
# @Time : 2021/2/1 14:00
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
import torch
import numpy as np
from torch.utils.data import DataLoader
from seqeval.metrics.sequence_labeling import get_entities
from collections import defaultdict


from params import dataset, BASE_MODEL_DIR, MAX_SEQ_LENGTH, VALID_BATCH_SIZE, ModelTokenizer
from model_train import BERTClass, CustomDataset, idx2tag

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BERTClass()
model.to(dev)
model.load_state_dict(torch.load("{}_{}_ner.pth".format(BASE_MODEL_DIR, dataset)))
tokenizer = ModelTokenizer.from_pretrained('../{}'.format(BASE_MODEL_DIR))


def get_model_tokens(text):
    code = tokenizer.encode(text)
    tokens = []
    for token in code[1:-1]:
        if tokenizer.convert_ids_to_tokens(token) == "[UNK]":
            tokens.append(tokenizer.convert_ids_to_tokens(token))
        else:
            tokens.append(tokenizer.convert_ids_to_tokens(token).replace("##", ""))
    return tokens


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
    test_text = "中新网兰州2月4日电 (记者 冯志军)甘肃省 生态环境厅4日发布消息称，今年甘肃将担好上游责任，深入推进黄河流域生态保护和污染防治，聚焦黄河流域突出生态环境问题和生态环境领域基础设施短板弱项，谋划实施一批生态保护、污染防治、能力建设等重大项目。".lower()
    tokens = get_model_tokens(test_text)
    tags = get_text_predict(test_text)
    seq_entity_list = get_entities(tags)
    print(seq_entity_list)
    entity_dict = defaultdict(list)
    for entity in seq_entity_list:
        entity_type, entity_start, entity_end = entity
        if entity_start < len(tokens):
            word = "".join(tokens[entity_start:entity_end+1])
            entity_dict[entity_type].append({"word": word,
                                             "start_index": test_text.find(word)
                                            })

    from pprint import pprint
    pprint(entity_dict)