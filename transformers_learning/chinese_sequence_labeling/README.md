本项目采用PyTorch框架，结合transformers和bert-base-chinese、ernie-1.0模型实现中文序列标注。

### 维护者

- jclian91

### 数据集

0. MSRA数据集（msra.train 条数据和msra.test 条数据），共3种标签：地点（LOC）, 人名（PER）, 组织机构（ORG）
1. 人民日报命名实体识别数据集（example.train 28046条数据和example.test 4636条数据），共3种标签：地点（LOC）, 人名（PER）, 组织机构（ORG）
2. 时间识别数据集（time.train 1700条数据和time.test 300条数据），共1种标签：TIME
3. CLUENER细粒度实体识别数据集（cluener.train 10748条数据和cluener.test 1343条数据），共10种标签：地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）


### 模型结构

transformers中的BertForTokenClassification模型

### 模型效果

#### bert-base-chinese模型

- MSRA数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=8

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

         LOC     0.9511    0.9409    0.9460      2876
         ORG     0.8576    0.9324    0.8934      1331
         PER     0.9554    0.9496    0.9525      1983

   micro avg     0.9309    0.9418    0.9363      6190
   macro avg     0.9214    0.9409    0.9306      6190
weighted avg     0.9324    0.9418    0.9368      6190
```

- 人民日报命名实体识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=8

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

         LOC     0.9456    0.9076    0.9262      3658
         ORG     0.8774    0.9076    0.8922      2185
         PER     0.9743    0.9549    0.9645      1864

   micro avg     0.9322    0.9190    0.9256      7707
   macro avg     0.9324    0.9234    0.9277      7707
weighted avg     0.9332    0.9190    0.9258      7707
```

- 时间识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=16, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

        TIME     0.8480    0.8980    0.8722       441

   micro avg     0.8480    0.8980    0.8722       441
   macro avg     0.8480    0.8980    0.8722       441
weighted avg     0.8480    0.8980    0.8722       441
```

- CLUENER细粒度实体识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=8

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

     address     0.5345    0.5962    0.5636       364
        book     0.7500    0.7895    0.7692       152
     company     0.7379    0.7923    0.7642       366
        game     0.8058    0.8676    0.8356       287
  government     0.7083    0.8361    0.7669       244
       movie     0.8322    0.7933    0.8123       150
        name     0.8376    0.8692    0.8531       451
organization     0.6768    0.7791    0.7243       344
    position     0.7025    0.7835    0.7408       425
       scene     0.6650    0.6683    0.6667       199

   micro avg     0.7183    0.7797    0.7477      2982
   macro avg     0.7251    0.7775    0.7497      2982
weighted avg     0.7207    0.7797    0.7485      2982
```

#### ernie-1.0模型

- MSRA数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=8

运行model_evaluate.py,模型评估结果如下：

```

```

- 人民日报命名实体识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=8

运行model_evaluate.py,模型评估结果如下：

```

```

- 时间识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=16, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

        TIME     0.8319    0.8866    0.8584       441

   micro avg     0.8319    0.8866    0.8584       441
   macro avg     0.8319    0.8866    0.8584       441
weighted avg     0.8319    0.8866    0.8584       441
```

- CLUENER细粒度实体识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=8

运行model_evaluate.py,模型评估结果如下：

```

```

#### Roberta-Chinese-Large模型

- MSRA数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=8

运行model_evaluate.py,模型评估结果如下：

```

```

- 人民日报命名实体识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=8

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

         LOC     0.9303    0.9122    0.9212      3658
         ORG     0.9058    0.9021    0.9039      2185
         PER     0.9674    0.9549    0.9611      1864

   micro avg     0.9323    0.9197    0.9259      7707
   macro avg     0.9345    0.9231    0.9287      7707
weighted avg     0.9323    0.9197    0.9260      7707
```

- 时间识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=16, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

        TIME     0.8571    0.8980    0.8771       441

   micro avg     0.8571    0.8980    0.8771       441
   macro avg     0.8571    0.8980    0.8771       441
weighted avg     0.8571    0.8980    0.8771       441
```

- CLUENER细粒度实体识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=8

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

     address     0.5733    0.6016    0.5871       364
        book     0.7561    0.8158    0.7848       152
     company     0.7456    0.8169    0.7797       366
        game     0.7812    0.8711    0.8237       287
  government     0.7605    0.8197    0.7890       244
       movie     0.7881    0.7933    0.7907       150
        name     0.8722    0.8780    0.8751       451
organization     0.7143    0.7994    0.7545       344
    position     0.7294    0.8118    0.7684       425
       scene     0.6789    0.7437    0.7098       199

   micro avg     0.7396    0.7964    0.7670      2982
   macro avg     0.7400    0.7951    0.7663      2982
weighted avg     0.7407    0.7964    0.7672      2982
```

### 模型预测示例

- 人民日报命名实体识别数据集

运行model_predict.py，对新文本进行预测，结果如下：

```
string: 当2016年6月24日凌晨，“脱欧”公投的最后一张选票计算完毕，占投票总数52%的支持选票最终让英国开始了一段长达4年的“脱欧”进程，其间卡梅伦、特雷莎·梅相继离任，“脱欧”最终在第三位首相鲍里斯·约翰逊任内完成。
entities:

{'LOC': [{'index': 48, 'name': '英国'}],
 'PER': [{'index': 69, 'name': '卡梅伦'},
         {'index': 73, 'name': '特雷莎·梅'},
         {'index': 95, 'name': '鲍里斯·约翰逊'}]}
```

```
string: 台湾“立法院”“莱猪（含莱克多巴胺的猪肉）”表决大战落幕，台当局领导人蔡英文24日晚在脸书发文宣称，“开放市场的决定，将会是未来台湾国际经贸走向世界的关键决定”。
entities:

{'LOC': [{'index': 0, 'name': '台湾'},
         {'index': 29, 'name': '台'},
         {'index': 64, 'name': '台湾'}],
 'PER': [{'index': 35, 'name': '蔡英文'}]}
```

```
string: 最近一段时间，印度政府在南海问题上接连发声。在近期印度、越南两国举行的线上总理峰会上，印度总理莫迪声称南海行为准则“不应损害该地区其他国家或第三方的利益”，两国总理还强调了所谓南海“航行自由”的重要性。
entities:

{'LOC': [{'index': 7, 'name': '印度'},
         {'index': 12, 'name': '南海'},
         {'index': 25, 'name': '印度'},
         {'index': 28, 'name': '越南'},
         {'index': 43, 'name': '印度'},
         {'index': 51, 'name': '南海'},
         {'index': 88, 'name': '南海'}],
 'PER': [{'index': 47, 'name': '莫迪'}]}
```

- 时间识别数据集

运行model_predict.py，对新文本进行预测，结果如下：

```
string: 去年11月30日，李先生来到茶店子东街一家银行取钱，准备购买家具。输入密码后，
entities:

{'TIME': [{'index': 0, 'name': '去年11月30日'}]}
```

```
string: 苏北大量农村住房建于上世纪80年代之前。去年9月，江苏省决定全面改善苏北农民住房条件，计划3年内改善30万户，作为决胜全面建成小康社会补短板的重要举措。
entities:

{'TIME': [{'index': 10, 'name': '上世纪80年代之前'},
          {'index': 20, 'name': '去年9月'},
          {'index': 45, 'name': '3年'}]}
```

```
string: 经过工作人员两天的反复验证、严密测算，记者昨天从上海中心大厦得到确认：被誉为上海中心大厦“定楼神器”的阻尼器，在8月10日出现自2016年正式启用以来的最大摆幅。
entities:

{'TIME': [{'index': 6, 'name': '两天'},
          {'index': 21, 'name': '昨天'},
          {'index': 56, 'name': '8月10日'},
          {'index': 64, 'name': '2016年'}]}
```

- CLUENER细粒度实体识别数据集

运行model_predict.py，对新文本进行预测，结果如下：

```
{'entities': [{'end': 5, 'start': 0, 'type': 'organization', 'word': '四川敦煌学'},
              {'end': 13, 'start': 11, 'type': 'scene', 'word': '丹棱'},
              {'end': 44, 'start': 41, 'type': 'name', 'word': '胡文和'}],
 'string': '四川敦煌学”。近年来，丹棱县等地一些不知名的石窟迎来了海内外的游客，他们随身携带着胡文和的著作。'}
```

```
{'entities': [{'end': 19, 'start': 14, 'type': 'address', 'word': '茶店子东街'}],
 'string': '去年11月30日，李先生来到茶店子东街一家银行取钱，准备购买家具。输入密码后，'}
```

```
{'entities': [{'end': 3, 'start': 0, 'type': 'name', 'word': '罗伯茨'},
              {'end': 10, 'start': 4, 'type': 'movie', 'word': '《逃跑新娘》'},
              {'end': 23, 'start': 16, 'type': 'movie', 'word': '《理发师佐翰》'},
              {'end': 38, 'start': 32, 'type': 'name', 'word': '亚当·桑德勒'}],
 'string': '罗伯茨的《逃跑新娘》不相伯仲；而《理发师佐翰》让近年来顺风顺水的亚当·桑德勒首尝冲过1亿＄'}
```


### 代码说明

0. 将transformers的BERT中文预训练模型bert-base-chinese放在bert-base-chinese文件夹下
1. 运行load_data.py，生成类别标签，注意O标签为0;
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/example.train和data/example.test的格式准备好
4. 调整模型参数，运行model_train.py进行模型训练
5. 运行model_evaluate.py进行模型评估
6. 运行model_predict.py对新文本进行预测