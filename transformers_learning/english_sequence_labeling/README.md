本项目采用PyTorch和transformers模块实现英语序列标注，其中对BERT进行微调。

### 维护者

- jclian91

### 数据集

1. [Conll2003](https://www.clips.uantwerpen.be/conll2003/ner/)

    conll2003.train 14987条数据和conll2003.test 3466条数据，共4种标签：
    
    + [x] LOC
    + [x] PER
    + [x] ORG
    + [x] MISC
    
2. [wnut17](https://noisy-text.github.io/2017/emerging-rare-entities.html)

    wnut17.train 3394条数据和wnut17.test 1009条数据，共6种标签：
    
    + [x] Person
    + [x] Location (including GPE, facility)
    + [x] Corporation
    + [x] Consumer good (tangible goods, or well-defined services)
    + [x] Creative work (song, movie, book, and so on)
    + [x] Group (subsuming music band, sports team, and non-corporate organisations)

### 模型结构

transformers中的BertForTokenClassification模型

### 模型效果

- Conll2003

模型参数：bert-base-uncased, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

         LOC     0.9444    0.9706    0.9573      1837
        MISC     0.8579    0.8709    0.8644       922
         ORG     0.8993    0.9128    0.9060      1341
         PER     0.9772    0.9794    0.9783      1842

   micro avg     0.9309    0.9448    0.9378      5942
   macro avg     0.9197    0.9334    0.9265      5942
weighted avg     0.9310    0.9448    0.9378      5942
```

[最新SOTA结果的F1值为94.3%.](https://github.com/sebastianruder/NLP-progress/blob/master/english/named_entity_recognition.md)

- wnut17

模型参数：bert-base-uncased, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

运行model_evaluate.py,模型评估结果如下：

```
               precision    recall  f1-score   support

  corporation     0.2667    0.3529    0.3038        34
creative-work     0.2500    0.1333    0.1739       105
        group     0.2059    0.1795    0.1918        39
     location     0.5250    0.5676    0.5455        74
       person     0.7711    0.6809    0.7232       470
      product     0.5263    0.1754    0.2632       114

    micro avg     0.6213    0.4964    0.5519       836
    macro avg     0.4242    0.3483    0.3669       836
 weighted avg     0.6036    0.4964    0.5339       836
```


### 代码说明

0. 将hugging face的bert-base-uncased预训练模型放在对应的文件夹下
1. 运行load_data.py，生成类别标签文件label2id.json，注意O标签为0;
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/conll2003.train和data/conll2003.test的格式准备好
4. 调整模型参数，运行torch_model_train.py进行模型训练
5. 运行torch_model_evaluate.py进行模型评估
6. 运行torch_model_predict.py对新文本进行预测