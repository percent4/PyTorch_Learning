# CAPTCHA recognizition in PyTorch

Using CNN model to recognize CAPTCHA by using PyTorch.

### dataset

1988 captcha images.

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
epoch: 99, batch: 87, total loss: 5.844647407531738, loss1: 1.461167573928833, loss2: 1.4611585140228271, loss3: 1.46117103099823, loss4: 1.4611504077911377
epoch: 99, batch: 88, total loss: 5.844733715057373, loss1: 1.4611550569534302, loss2: 1.4611501693725586, loss3: 1.461153268814087, loss4: 1.4612756967544556
epoch: 99, batch: 89, total loss: 5.84473180770874, loss1: 1.4611501693725586, loss2: 1.4611537456512451, loss3: 1.4612778425216675, loss4: 1.461150050163269
epoch: 99, batch: 90, total loss: 5.844989776611328, loss1: 1.4611608982086182, loss2: 1.461156964302063, loss3: 1.461521029472351, loss4: 1.4611506462097168
epoch: 99, batch: 91, total loss: 5.84489631652832, loss1: 1.4611504077911377, loss2: 1.4611639976501465, loss3: 1.4614293575286865, loss4: 1.4611525535583496
epoch: 99, batch: 92, total loss: 5.844632625579834, loss1: 1.4611501693725586, loss2: 1.4611501693725586, loss3: 1.4611589908599854, loss4: 1.4611735343933105
epoch: 99, batch: 93, total loss: 5.844607353210449, loss1: 1.4611515998840332, loss2: 1.4611501693725586, loss3: 1.4611552953720093, loss4: 1.4611499309539795
epoch: 99, batch: 94, total loss: 5.844761848449707, loss1: 1.4612501859664917, loss2: 1.4611870050430298, loss3: 1.461174488067627, loss4: 1.4611504077911377
epoch: 99, batch: 95, total loss: 5.8446245193481445, loss1: 1.4611499309539795, loss2: 1.4611501693725586, loss3: 1.4611742496490479, loss4: 1.4611504077911377
epoch: 99, batch: 96, total loss: 5.844852924346924, loss1: 1.4611520767211914, loss2: 1.461226463317871, loss3: 1.4613206386566162, loss4: 1.4611536264419556
epoch: 99, batch: 97, total loss: 5.876787185668945, loss1: 1.4634206295013428, loss2: 1.462177038192749, loss3: 1.4900226593017578, loss4: 1.4611670970916748
epoch: 99, batch: 98, total loss: 5.847787380218506, loss1: 1.4611519575119019, loss2: 1.4611936807632446, loss3: 1.464290976524353, loss4: 1.4611506462097168
epoch: 99, batch: 99, total loss: 5.844601154327393, loss1: 1.4611501693725586, loss2: 1.4611502885818481, loss3: 1.4611505270004272, loss4: 1.4611501693725586
Accuracy 1: 0.9975, Accuracy 2: 0.9874, Accuracy 3: 0.9849, Accuracy 4: 0.9799
```

### model evaluate

Accuracy 1: 0.9975, Accuracy 2: 0.9874, Accuracy 3: 0.9849, Accuracy 4: 0.9799

### model predict on new captcha image

no characters are wrong, total 48 characters.

### references

1. CAPTCHA-Recognizition: https://github.com/percent4/CAPTCHA-Recognizition

