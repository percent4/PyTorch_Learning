`Datasets`库是`HuggingFace`生态系统中一个重要的数据集库，可用于轻松地访问和共享数据集，这些数据集是关于音频、计算机视觉、以及自然语言处理等领域。`Datasets` 库可以通过一行来加载一个数据集，并且可以使用 `Hugging Face` 强大的数据处理方法来快速准备好你的数据集。在 `Apache Arrow` 格式的支持下，通过 `zero-copy read` 来处理大型数据集，而没有任何内存限制，从而实现最佳速度和效率。

当需要微调模型的时候，需要对数据集进行以下操作：

1.  数据集加载：下载、加载数据集&#x20;
2.  数据集预处理：使用Dataset.map() 预处理数据&#x20;
3.  数据集评估指标：加载和计算指标

可以在HuggingFace官网来搜共享索数据集：<https://huggingface.co/datasets​> 。本文中使用的主要数据集为`squad`数据集，其在HuggingFace网站上的数据前几行如下：

### 加载数据

*   加载Dataset数据集

Dataset数据集可以是HuggingFace Datasets网站上的数据集或者是本地路径对应的数据集，也可以同时加载多个数据集。

以下是加载英语阅读理解数据集`squad`， 该数据集的网址为：<https://huggingface.co/datasets/squad> ，也是本文中使用的主要数据集。

```python
import datasets

# 加载单个数据集
raw_datasets = datasets.load_dataset('squad')
# 加载多个数据集
raw_datasets = datasets.load_dataset('glue', 'mrpc')
```

*   从文件中加载数据

支持csv, tsv, txt, json, jsonl等格式的文件

```python
from datasets import load_dataset

data_files = {"train": "./data/sougou_mini/train.csv", "test": "./data/sougou_mini/test.csv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter=",")
```

*   从Dataframe中加载数据

```python
import pandas as pd
from datasets import Dataset 

my_dict = {"a": [1, 2, 3], "b": ['A', 'B', 'C']} 
dataset1 = Dataset.from_dict(my_dict) 
 
df = pd.DataFrame(my_dict) 
dataset2 = Dataset.from_pandas(df)
```

### &#x20;查看数据

*   数据结构

数据结构包括：

*   数据集的划分：train，valid，test数据集
*   数据集的数量
*   数据集的feature

`squad`数据的数据结构如下：

```json
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 10570
    })
})
```

*   数据切分

```python
import datasets

raw_dataset = datasets.load_dataset('squad')

# 获取某个划分数据集，比如train
train_dataset = raw_dataset['train']
# 获取前10条数据
head_dataset = train_dataset.select(range(10))
# 获取随机10条数据
shuffle_dataset = train_dataset.shuffle(seed=42).select(range(10))
# 数据切片
slice_dataset = train_dataset[10:20]
```

### 更多特性

*   数据打乱（shuffle）

shuffle的功能是打乱datasets中的数据，其中seed是设置打乱的参数，如果设置打乱的seed是相同的，那我们就可以得到一个完全相同的打乱结果，这样用相同的打乱结果才能重复的进行模型试验。

```python
import datasets

raw_dataset = datasets.load_dataset('squad')
# 打乱数据集
shuffle_dataset = train_dataset.shuffle(seed=42)
```

*   数据流（stream）

stream的功能是将数据集进行流式化，可以不用在下载整个数据集的情况下使用该数据集。这在以下场景中特别有用：

1.  你不想等待整个庞大的数据集下载完毕
2.  数据集大小超过了你计算机的可用硬盘空间
3.  你想快速探索数据集的少数样本

```python
from datasets import load_dataset

dataset = load_dataset('oscar-corpus/OSCAR-2201', 'en', split='train', streaming=True)
print(next(iter(dataset)))
```

*   数据列重命名（rename columns）

数据集支持对列重命名。下面的代码将`squad`数据集中的context列重命名为text：

```python
from datasets import load_dataset

squad = load_dataset('squad')
squad = squad.rename_column('context', 'text')
```

*   数据丢弃列（drop columns）

数据集支持对列进行丢弃，在删除一个或多个列时，向`remove_columns()`函数提供要删除的列名。单个列删除传入列名，多个列删除传入列名的列表。下面的代码将`squad`数据集中的id列丢弃：

```python
from datasets import load_dataset

squad = load_dataset('squad')
# 删除一个列
squad = squad.remove_columns('id')
# 删除多个列
squad = squad.remove_columns(['title', 'text'])
```

*   数据新增列（add new columns）

数据集支持新增列。下面的代码在`squad`数据集上新增一列test，内容全为字符串111：

```python
from datasets import load_dataset

squad = load_dataset('squad')
# 新增列
new_train_squad = squad['train'].add_column("test", ['111'] * squad['train'].num_rows)
```

*   数据类型转换（cast）

`cast()`函数对一个或多个列的特征类型进行转换。这个函数接受你的新特征作为其参数。

```python
from datasets import load_dataset

squad = load_dataset('squad')
# 新增列
new_train_squad = squad['train'].add_column("test", ['111'] * squad['train'].num_rows)
print(new_train_squad.features)
# 转换test列的数据类型
new_features = new_train_squad.features.copy()
new_features["test"] = Value("int64")
new_train_squad = new_train_squad.cast(new_features)
# 输出转换后的数据类型
print(new_train_squad.features)
```

*   数据展平（flatten）

针对嵌套结构的数据类型，可使用`flatten()`函数将子字段提取到它们自己的独立列中。

```python
from datasets import load_dataset

squad = load_dataset('squad')
flatten_dataset = squad['train'].flatten()
print(flatten_dataset)
```

输出结果为：

```bash
Dataset({
    features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
    num_rows: 87599
})
```

*   数据合并（Concatenate Multiple Datasets）

如果独立的数据集有相同的列类型，那么它们可以被串联起来。用`concatenate_datasets()`来连接不同的数据集。

```python
from datasets import concatenate_datasets, load_dataset

squad = load_dataset('squad')
squad_v2 = load_dataset('squad_v2')
# 合并数据集
squad_all = concatenate_datasets([squad['train'], squad_v2['train']])
```

*   数据过滤（filter）

`filter()`函数支持对数据集进行过滤，一般采用lambda函数实现。下面的代码对`squad`数据集中的训练集的question字段，过滤掉split后长度小于等于10的数据：

```python
from datasets import load_dataset

squad = load_dataset('squad')
filter_dataset = squad['train'].filter(lambda x: len(x["question"].split()) > 10)
```

输出结果如下：

```bash
Dataset({
    features: ['id', 'title', 'context', 'question', 'answers'],
    num_rows: 34261
})
```

*   数据排序（sort）

使用`sort()`对列值根据其数值进行排序。下面的代码是对`squad`数据集中的训练集按照标题长度进行排序：

```python
from datasets import load_dataset

squad = load_dataset('squad')
# 新增列, title_length, 标题长度
new_train_squad = squad['train'].add_column("title_length", [len(_) for _ in squad['train']['title']])
# 按照title_length排序
new_train_squad = new_train_squad.sort("title_length")
```

*   数据格式（set\_format）

`set_format()`函数改变了一个列的格式，使之与一些常见的数据格式兼容。在类型参数中指定你想要的输出和你想要格式化的列。格式化是即时应用的。支持的数据格式有：None, numpy, torch, tensorflow, pandas, arrow,  如果选择None，就会返回python对象。

下面的代码将新增标题长度列，并将其转化为numpy格式：

```python
from datasets import load_dataset

squad = load_dataset('squad')
# 新增列, title_length, 标题长度
new_train_squad = squad['train'].add_column("title_length", [len(_) for _ in squad['train']['title']])
# 转换为numpy支持的数据格式
new_train_squad.set_format(type="numpy", columns=["title_length"])
```

*   数据指标（load metrics）

[HuggingFace Hub](https://huggingface.co/metrics)上提供了一系列的评估指标（metrics），前20个指标如下：

```python
from datasets import list_metrics
metrics_list = list_metrics()
print(', '.join(metric for metric in metrics_list[:20]))
```

输出结果如下：

```bash
accuracy, bertscore, bleu, bleurt, brier_score, cer, character, charcut_mt, chrf, code_eval, comet, competition_math, coval, cuad, exact_match, f1, frugalscore, glue, google_bleu, indic_glue
```

从Hub中加载一个指标，使用 [`datasets.load_metric()`](https://huggingface.co/docs/datasets/v1.0.1/package_reference/loading_methods.html#datasets.load_metric "datasets.load_metric") 命令，比如加载`squad`数据集的指标：

```python
from datasets import load_metric
metric = load_metric('squad')
```

输出结果如下：

```bash
Metric(name: "squad", features: {'predictions': {'id': Value(dtype='string', id=None), 'prediction_text': Value(dtype='string', id=None)}, 'references': {'id': Value(dtype='string', id=None), 'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None)}}, usage: """
Computes SQuAD scores (F1 and EM).
Args:
    predictions: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair as given in the references (see below)
        - 'prediction_text': the text of the answer
    references: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair (see above),
        - 'answers': a Dict in the SQuAD dataset format
            {
                'text': list of possible texts for the answer, as a list of strings
                'answer_start': list of start positions for the answer, as a list of ints
            }
            Note that answer_start values are not taken into account to compute the metric.
Returns:
    'exact_match': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
Examples:

    >>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
    >>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
    >>> squad_metric = datasets.load_metric("squad")
    >>> results = squad_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'exact_match': 100.0, 'f1': 100.0}
""", stored examples: 0)
```

load\_metric还支持分布式计算，本文不再详细讲述。

load\_metric现在已经是老版本了，新版本将用`evaluate`模块代替，访问网址为：<https://github.com/huggingface/evaluate> 。

*   数据映射（map）

map就是映射，它接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset。常见的map函数的应用是对文本进行tokenize：

```python
from datasets import load_dataset
from transformers import AutoTokenizer

squad_dataset = load_dataset('squad')

checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(sample):
    return tokenizer(sample['context'], truncation=True, max_length=256)

tokenized_dataset = squad_dataset.map(tokenize_function, batched=True)
```

输出结果如下：

```bash
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 87599
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 10570
    })
})
```

*   数据保存/加载（save to disk/ load from disk）

使用`save_to_disk()`来保存数据集，方便在以后重新使用它,使用 `load_from_disk()`函数重新加载数据集。我们将上面map后的tokenized\_dataset数据集进行保存：

```python
tokenized_dataset.save_to_disk("squad_tokenized")
```

保存后的文件结构如下：

```bash
squad_tokenized/
├── dataset_dict.json
├── train
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
└── validation
    ├── data-00000-of-00001.arrow
    ├── dataset_info.json
    └── state.json
```

加载数据的代码如下：

```python
from datasets import load_from_disk
reloaded_dataset = load_from_disk("squad_tokenized") 
```

### 总结

本文可作为dataset库的入门，详细介绍数据集的各种操作，这样方便后续进行模型训练。

### &#x20;参考文献

1.  Datasets: <https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/2_datasets.html>
2.  Huggingface详细入门介绍之dataset库：<https://zhuanlan.zhihu.com/p/554678463>
3.  Stream: <https://huggingface.co/docs/datasets/stream>
4.  HuggingFace教程 Datasets基本操作: Process: <https://zhuanlan.zhihu.com/p/557032513>

