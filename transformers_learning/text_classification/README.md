a本项目采用PyTorch和Transformers模块实现文本多分类任务。

### 维护者

- jclian91

### 数据集

#### sougou小分类数据集

sougou小分类数据集，共有5个类别，分别为体育、健康、军事、教育、汽车。

划分为训练集和测试集，其中训练集每个分类800条样本，测试集每个分类100条样本。


### 代码结构

```
.
├── data（数据集）
│   └── sougou_mini
│       ├── test.csv
│       └── train.csv
├── __init__.py
├── model_predict.py（模型预测脚本）
├── model_train.py（模型训练脚本）
├── params.py（参数配置文件）
├── sougou_mini_cls.pth（模型保存文件）
└── sougou_mini_label2id.json（分类类别文件）
```

## 模型效果

#### sougou数据集

模型: bert-base-chinese
模型参数: batch_size = 32, maxlen = 256, epoch=10

评估结果:

```
              precision    recall  f1-score   support

          体育     0.9900    1.0000    0.9950        99
          健康     0.9700    0.9798    0.9749        99
          军事     0.9802    1.0000    0.9900        99
          教育     0.9495    0.9495    0.9495        99
          汽车     1.0000    0.9596    0.9794        99

    accuracy                         0.9778       495
   macro avg     0.9779    0.9778    0.9777       495
weighted avg     0.9779    0.9778    0.9777       495
```

模型: ernie-1.0
模型参数: batch_size = 32, maxlen = 256, epoch=10

评估结果:

```
              precision    recall  f1-score   support

          体育     0.9900    1.0000    0.9950        99
          健康     0.9600    0.9697    0.9648        99
          军事     0.9900    1.0000    0.9950        99
          教育     0.9691    0.9495    0.9592        99
          汽车     0.9796    0.9697    0.9746        99

    accuracy                         0.9778       495
   macro avg     0.9777    0.9778    0.9777       495
weighted avg     0.9777    0.9778    0.9777       495
```

### 项目启动

1. 将transformers中的BERT中文预训练模型bert-base-chinese放在bert-base-chinese文件夹下
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/sougou_mini的格式准备好
4. 调整模型参数，运行model_train.py进行模型训练
5. 运行model_evaluate.py进行模型评估
6. 运行model_predict.py对新文本进行评估