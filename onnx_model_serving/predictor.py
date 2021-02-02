# -*- coding: utf-8 -*-
# @Time : 2021/1/29 17:08
# @Author : Jclian91
# @File : predictor.py
# @Place : Yangpu, Shanghai
# this is an example for cortex release 0.18 and may not deploy correctly on other releases of cortex

import torch


from model import IrisNet

labels = ["setosa", "versicolor", "virginica"]


class PythonPredictor:
    def __init__(self):
        # initialize the model
        model = IrisNet()
        model.load_state_dict(torch.load("iris.pth"))
        model.eval()

        self.model = model

    def predict(self, payload):
        # Convert the request to a tensor and pass it into the model
        input_tensor = torch.FloatTensor(
            [
                [
                    payload["sepal_length"],
                    payload["sepal_width"],
                    payload["petal_length"],
                    payload["petal_width"],
                ]
            ]
        )

        # Run the prediction
        output = self.model(input_tensor)

        # Translate the model output to the corresponding label string
        return labels[torch.argmax(output[0])]