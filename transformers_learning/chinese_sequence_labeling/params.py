# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from transformers import BertForTokenClassification, BertTokenizer, BertConfig
from transformers import RobertaForTokenClassification, RobertaTokenizer, RobertaConfig

# 数据相关的配置
dataset = "time"

train_file_path = "./data/%s.train" % dataset
test_file_path = "./data/%s.test" % dataset

# 模型选择
BASE_MODEL_DIR = "chinese-roberta-wwm-ext-large"
MODEL_MODE = "ROBERTA"
MODEL_CLASS = {"AUTO": {"model": AutoModelForTokenClassification, "config": AutoConfig, "tokenizer": AutoTokenizer},
               "BERT": {"model": BertForTokenClassification, "config": BertConfig, "tokenizer": BertTokenizer},
               "ROBERTA": {"model": BertForTokenClassification, "config": BertConfig, "tokenizer": BertTokenizer}
               }
assert MODEL_MODE in MODEL_CLASS.keys()
ModelTokenizer = MODEL_CLASS[MODEL_MODE]["tokenizer"]
ModelConfig = MODEL_CLASS[MODEL_MODE]["config"]
ModelForTokenClassification = MODEL_CLASS[MODEL_MODE]["model"]

# 模型配置
MAX_SEQ_LENGTH = 128   # 输入的文本最大长度
TRAIN_BATCH_SIZE = 16  # 模型训练的BATCH SIZE
VALID_BATCH_SIZE = 16
EPOCHS = 10            # 模型训练的轮次
LEARNING_RATE = 1e-5
