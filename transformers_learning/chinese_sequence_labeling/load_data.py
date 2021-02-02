# -*- coding: utf-8 -*-
# @Time : 2020/12/24 13:27
# @Author : Jclian91
# @File : load_data.py
# @Place : Yangpu, Shanghai
import json
from transformers import AutoTokenizer

from params import train_file_path, dataset, BASE_MODEL_DIR

tokenizer = AutoTokenizer.from_pretrained("../{}".format(BASE_MODEL_DIR))


def get_model_tags(text, true_tag):
    code = tokenizer.encode(text)
    model_tag = []
    tag_cnt = 0
    for token in code[1:-1]:
        model_tag.append(true_tag[tag_cnt])
        if tokenizer.convert_ids_to_tokens(token) == "[UNK]":
            tag_cnt += 1
        else:
            tag_cnt += len(tokenizer.convert_ids_to_tokens(token).replace("##", ""))
    return model_tag


# 读取数据集
def read_data(file_path):
    # 读取数据集
    with open(file_path, "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    # 读取空行所在的行号
    index = [-1]
    index.extend([i for i, _ in enumerate(content) if " " not in _])
    index.append(len(content))

    # 按空行分割，读取原文句子及标注序列
    sentences, tags = [], []
    for j in range(len(index)-1):
        sent, tag = [], []
        segment = content[index[j]+1: index[j+1]]
        for line in segment:
            sent.append(line.split(" ")[0])
            tag.append(line.split(" ")[-1])

        true_tag = get_model_tags(''.join(sent).lower(), tag)
        sentences.append(''.join(sent).lower())
        tags.append(true_tag)

    # 去除空的句子及标注序列，一般放在末尾
    sentences = [_ for _ in sentences if _]
    tags = [_ for _ in tags if _]

    return sentences, tags


# 读取训练集数据
# 将标签转换成id
def label2id():

    _, train_tags = read_data(train_file_path)

    # 标签转换成id，并保存成文件
    unique_tags = []
    for seq in train_tags:
        for _ in seq:
            if _ not in unique_tags and _ != "O":
                unique_tags.append(_)

    label_id_dict = {"O": 0}
    label_id_dict.update(dict(zip(unique_tags, range(1, len(unique_tags)+1))))

    with open("%s_label2id.json" % dataset, "w", encoding="utf-8") as g:
        g.write(json.dumps(label_id_dict, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sentences, tags = read_data(train_file_path)
    for sent, tag in zip(sentences[:10], tags[:10]):
        print(sent, tag)
    label2id()
