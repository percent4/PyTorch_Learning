# -*- coding: utf-8 -*-
# @Time : 2021/1/31 21:35
# @Author : Jclian91
# @File : torch_model_evaluate.py
# @Place : Yangpu, Shanghai
# 利用seqeval模块对序列标注的结果进行评估
from seqeval.metrics import classification_report

from util import test_file_path
from torch_model_predict import get_text_predict


if __name__ == '__main__':
    # 读取测试集数据
    with open(test_file_path, "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    # 读取空行所在的行号
    index = [-1]
    index.extend([i for i, _ in enumerate(content) if not _])
    index.append(len(content))

    # 按空行分割，读取原文句子及标注序列
    sentences, tags = [], []
    for j in range(len(index) - 1):
        sent, tag = [], []
        segment = content[index[j] + 1: index[j + 1]]
        for line in segment:
            word, bio_tag = line.split()[0], line.split()[-1]
            sent.append(word)
            tag.append(bio_tag)

        sentences.append(" ".join(sent))
        tags.append(tag)

    # 去除空的句子及标注序列，一般放在末尾
    input_test = [_ for _ in sentences if _]
    result_test = [_ for _ in tags if _]

    for sent, tag in zip(input_test[:10], result_test[:10]):
        print(sent, tag)

    # 测试集
    i = 1
    true_tag_list = []
    pred_tag_list = []
    for test_text, true_tag in zip(input_test, result_test):
        print("Predict %d samples" % i)
        print("test text: ", test_text)
        pred_tag = get_text_predict(text=test_text)
        true_tag_list.append(true_tag)
        print("true tag: ", true_tag)
        print("pred tag: ", pred_tag)
        if len(true_tag) <= len(pred_tag):
            pred_tag_list.append(pred_tag[:len(true_tag)])
        else:
            pred_tag_list.append(pred_tag+["O"]*(len(true_tag)-len(pred_tag)))
        i += 1

    print(classification_report(true_tag_list, pred_tag_list, digits=4))