# CAPTCHA recognizition in PyTorch

Using CNN model to recognize CAPTCHA by using PyTorch.

### Code structure

```
.
├── captcha_recognize.pth（验证码识别的保存后的模型）
├── __init__.py
├── model_train.py
├── new_images（新的验证码图片）
├── README.md
├── torch_model_train.py（模型训练脚本）
├── torch_model_predict.py（模型训练脚本）
└── train_images（训练验证码，共944张）

3 directories, 6 files
```

### CNN model

```
CNNModel(
  (hidden1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act1): ReLU()
  (hidden2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act2): ReLU()
  (pool1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (hidden3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act3): ReLU()
  (hidden4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act4): ReLU()
  (pool2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (hidden5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act5): ReLU()
  (hidden6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act6): ReLU()
  (pool3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (hidden7): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act7): ReLU()
  (hidden8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act8): ReLU()
  (pool4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (hidden9): Linear(in_features=768, out_features=100, bias=True)
  (act9): ReLU()
  (hidden10): Linear(in_features=768, out_features=100, bias=True)
  (act10): ReLU()
  (hidden11): Linear(in_features=768, out_features=100, bias=True)
  (act11): ReLU()
  (hidden12): Linear(in_features=768, out_features=100, bias=True)
  (act12): ReLU()
  (hidden13): Linear(in_features=100, out_features=10, bias=True)
  (hidden14): Linear(in_features=100, out_features=10, bias=True)
  (hidden15): Linear(in_features=100, out_features=10, bias=True)
  (hidden16): Linear(in_features=100, out_features=10, bias=True)
  (act13): Softmax(dim=1)
  (act14): Softmax(dim=1)
  (act15): Softmax(dim=1)
  (act16): Softmax(dim=1)
)
```

### model train

```
epoch: 99, batch: 80, total loss: 5.844603061676025, loss1: 1.4611523151397705, loss2: 1.4611502885818481, loss3: 1.4611502885818481, loss4: 1.4611501693725586
epoch: 99, batch: 81, total loss: 5.844607830047607, loss1: 1.461155652999878, loss2: 1.4611501693725586, loss3: 1.4611517190933228, loss4: 1.4611501693725586
epoch: 99, batch: 82, total loss: 5.844608783721924, loss1: 1.4611506462097168, loss2: 1.4611501693725586, loss3: 1.4611579179763794, loss4: 1.4611502885818481
epoch: 99, batch: 83, total loss: 5.844610691070557, loss1: 1.4611504077911377, loss2: 1.4611501693725586, loss3: 1.461155891418457, loss4: 1.4611544609069824
epoch: 99, batch: 84, total loss: 5.844602584838867, loss1: 1.4611504077911377, loss2: 1.4611501693725586, loss3: 1.4611518383026123, loss4: 1.4611502885818481
epoch: 99, batch: 85, total loss: 5.844611644744873, loss1: 1.461151123046875, loss2: 1.4611502885818481, loss3: 1.4611585140228271, loss4: 1.4611515998840332
epoch: 99, batch: 86, total loss: 5.844606876373291, loss1: 1.4611533880233765, loss2: 1.4611501693725586, loss3: 1.4611529111862183, loss4: 1.4611501693725586
epoch: 99, batch: 87, total loss: 5.844610214233398, loss1: 1.4611506462097168, loss2: 1.4611525535583496, loss3: 1.4611568450927734, loss4: 1.4611504077911377
epoch: 99, batch: 88, total loss: 5.844616413116455, loss1: 1.4611504077911377, loss2: 1.461150050163269, loss3: 1.4611515998840332, loss4: 1.4611645936965942
epoch: 99, batch: 89, total loss: 5.844627380371094, loss1: 1.4611760377883911, loss2: 1.4611501693725586, loss3: 1.461151361465454, loss4: 1.4611501693725586
epoch: 99, batch: 90, total loss: 5.844608306884766, loss1: 1.4611505270004272, loss2: 1.4611502885818481, loss3: 1.4611507654190063, loss4: 1.461156964302063
epoch: 99, batch: 91, total loss: 5.844614028930664, loss1: 1.4611619710922241, loss2: 1.4611502885818481, loss3: 1.4611514806747437, loss4: 1.4611501693725586
epoch: 99, batch: 92, total loss: 5.844626426696777, loss1: 1.4611597061157227, loss2: 1.4611504077911377, loss3: 1.4611506462097168, loss4: 1.461165189743042
epoch: 99, batch: 93, total loss: 5.844621181488037, loss1: 1.4611626863479614, loss2: 1.4611552953720093, loss3: 1.4611530303955078, loss4: 1.4611501693725586
epoch: 99, batch: 94, total loss: 5.84462308883667, loss1: 1.4611506462097168, loss2: 1.4611501693725586, loss3: 1.4611722230911255, loss4: 1.4611502885818481
Accuracy 1: 0.9683, Accuracy 2: 0.9894, Accuracy 3: 0.9841, Accuracy 4: 0.9894
```

### model evaluate

Accuracy 1: 0.9683, Accuracy 2: 0.9894, Accuracy 3: 0.9841, Accuracy 4: 0.9894

### model predict on new captcha image

only 3 characters are wrong, total 48 characters.

### references

1. CAPTCHA-Recognizition: https://github.com/percent4/CAPTCHA-Recognizition

