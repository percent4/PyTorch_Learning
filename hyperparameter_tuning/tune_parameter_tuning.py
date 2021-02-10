# -*- coding: utf-8 -*-
# @Time : 2021/2/10 15:31
# @Author : Jclian91
# @File : tune_parameter_tuning.py
# @Place : Yangpu, Shanghai
import torch.optim as optim
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test


def train_mnist(config):
    train_loader, test_loader = get_data_loaders()
    print("train num: ", len(train_loader))
    print("test num: ", len(test_loader))
    model = ConvNet()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    for i in range(30):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        tune.report(mean_accuracy=acc)  # 添加的代码


# 添加如下代码
analysis = tune.run(
    train_mnist,
    num_samples=10,
    # Uncomment this to let each evaluation use 1 GPU
    # resources_per_trial={"CPU": 1, "GPU": 1},
    config={"lr": tune.grid_search([0.001, 0.01, 0.1])})

print("Best config: ", analysis.get_best_config(metric="mean_accuracy", mode="max"))

# 获取结果的 dataframe
df = analysis.dataframe()
print(df)
