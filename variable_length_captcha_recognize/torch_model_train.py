# -*- coding: utf-8 -*-
# @Time : 2021/1/26 12:04
# @Author : Jclian91
# @File : torch_model_train.py
# @Place : Yangpu, Shanghai
import os
import cv2
import numpy as np
import torch
from numpy import vstack
from numpy import argmax
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Softmax, Module, CrossEntropyLoss, Conv2d, MaxPool2d, BCELoss, Sigmoid
from torch.nn.init import kaiming_uniform_, xavier_uniform_

SIGNS = "+-"
CHARACTERS = '0123456789@'
IMAGE_WIDTH, IMAGE_HEIGHT, N_LENGTH, N_CLASS = 100, 30, 5, 13


# dataset definition
class CaptchaDataset(Dataset):
    # load the dataset
    def __init__(self):
        image_dir = "./train_images"
        image_num = len(os.listdir(image_dir))
        self.X = np.zeros((image_num, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        self.y1 = np.zeros(image_num, dtype=np.uint8)
        self.y2 = np.zeros(image_num, dtype=np.uint8)
        self.y3 = np.zeros(image_num, dtype=np.uint8)
        self.y4 = np.zeros(image_num, dtype=np.uint8)
        self.y5 = np.zeros(image_num, dtype=np.uint8)

        for i, file in enumerate(os.listdir(image_dir)):
            image_pixel = cv2.imread(os.path.join(image_dir, file), 1)/255
            self.X[i] = torch.from_numpy(image_pixel).permute(2, 0, 1).numpy()
            filename = file[:5]
            self.y1[i] = CHARACTERS.index(filename[0])
            self.y2[i] = CHARACTERS.index(filename[1])
            self.y3[i] = SIGNS.index(filename[2])
            self.y4[i] = CHARACTERS.index(filename[3])
            self.y5[i] = CHARACTERS.index(filename[4])

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y1[idx], self.y2[idx], self.y3[idx], self.y4[idx], self.y5[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class CNNModel(Module):
    # define model elements
    def __init__(self):
        super(CNNModel, self).__init__()
        # cnn module 1
        self.hidden1 = Conv2d(3, 32, (3, 3), padding=(1, 1))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Conv2d(32, 32, (3, 3), padding=(1, 1))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))
        # cnn module 2
        self.hidden3 = Conv2d(32, 32*2**1, (3, 3), padding=(1, 1))
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # second hidden layer
        self.hidden4 = Conv2d(32*2**1, 32*2**1, (3, 3), padding=(1, 1))
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))
        # cnn module 3
        self.hidden5 = Conv2d(32*2**1, 32*2**2, (3, 3), padding=(1, 1))
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        # second hidden layer
        self.hidden6 = Conv2d(32*2**2, 32*2**2, (3, 3), padding=(1, 1))
        kaiming_uniform_(self.hidden6.weight, nonlinearity='relu')
        self.act6 = ReLU()
        self.pool3 = MaxPool2d((2, 2), stride=(2, 2))
        # cnn module 4
        self.hidden7 = Conv2d(32*2**2, 32*2**3, (3, 3), padding=(1, 1))
        kaiming_uniform_(self.hidden7.weight, nonlinearity='relu')
        self.act7 = ReLU()
        # second hidden layer
        self.hidden8 = Conv2d(32*2**3, 32*2**3, (3, 3), padding=(1, 1))
        kaiming_uniform_(self.hidden8.weight, nonlinearity='relu')
        self.act8 = ReLU()
        self.pool4 = MaxPool2d((2, 2), stride=(2, 2))
        # fully connected layer
        self.hidden9 = Linear(256*1*6, 100)
        kaiming_uniform_(self.hidden9.weight, nonlinearity='relu')
        self.act9 = ReLU()
        self.hidden10 = Linear(256 * 1 * 6, 100)
        kaiming_uniform_(self.hidden10.weight, nonlinearity='relu')
        self.act10 = ReLU()
        self.hidden11 = Linear(256 * 1 * 6, 100)
        kaiming_uniform_(self.hidden11.weight, nonlinearity='relu')
        self.act11 = ReLU()
        self.hidden12 = Linear(256 * 1 * 6, 100)
        kaiming_uniform_(self.hidden12.weight, nonlinearity='relu')
        self.act12 = ReLU()
        self.hidden13 = Linear(256 * 1 * 6, 100)
        kaiming_uniform_(self.hidden13.weight, nonlinearity='relu')
        self.act13 = ReLU()
        # output layer
        self.hidden14 = Linear(100, 10)
        self.hidden15 = Linear(100, 10)
        self.hidden16 = Linear(100, 2)
        self.hidden17 = Linear(100, 10)
        self.hidden18 = Linear(100, 11)
        xavier_uniform_(self.hidden14.weight)
        xavier_uniform_(self.hidden15.weight)
        xavier_uniform_(self.hidden16.weight)
        xavier_uniform_(self.hidden17.weight)
        xavier_uniform_(self.hidden18.weight)
        self.act14 = Softmax(dim=1)
        self.act15 = Softmax(dim=1)
        self.act16 = Softmax(dim=1)
        self.act17 = Softmax(dim=1)
        self.act18 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # cnn module 1
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool1(X)
        # print("Pool 1: ", X.shape)
        # cnn module 2
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        X = self.act4(X)
        X = self.pool2(X)
        # print("Pool 2: ", X.shape)
        # cnn module 1
        X = self.hidden5(X)
        X = self.act5(X)
        X = self.hidden6(X)
        X = self.act6(X)
        X = self.pool3(X)
        # print("Pool 3: ", X.shape)
        # cnn module 4
        X = self.hidden7(X)
        X = self.act7(X)
        X = self.hidden8(X)
        X = self.act8(X)
        X = self.pool4(X)
        # print("Pool 4: ", X.shape)
        # flatten
        X = X.view(-1, 256*1*6)
        T = X.view(-1, 256*1*6)
        # output layer
        X1 = self.hidden9(X)
        X1 = self.hidden14(X1)
        X1 = self.act14(X1)
        X2 = self.hidden10(X)
        X2 = self.hidden15(X2)
        X2 = self.act15(X2)
        X3 = self.hidden11(T)
        X3 = self.hidden16(X3)
        X3 = self.act16(X3)
        X4 = self.hidden12(X)
        X4 = self.hidden17(X4)
        X4 = self.act17(X4)
        X5 = self.hidden13(X)
        X5 = self.hidden18(X5)
        X5 = self.act18(X5)
        return X1, X2, X3, X4, X5


# prepare the dataset
def prepare_data():
    # load the dataset
    dataset = CaptchaDataset()
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_data = DataLoader(train, batch_size=16, shuffle=True)
    test_data = DataLoader(test, batch_size=1024, shuffle=False)
    return train_data, test_data


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion1 = CrossEntropyLoss()
    criterion2 = CrossEntropyLoss()
    criterion3 = CrossEntropyLoss()
    criterion4 = CrossEntropyLoss()
    criterion5 = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, target_y1, target_y2, target_y3, target_y4, target_y5) in enumerate(train_dl):
            # inputs = inputs.cuda()
            target_y1 = target_y1.long()
            target_y2 = target_y2.long()
            target_y3 = target_y3.long()
            target_y4 = target_y4.long()
            target_y5 = target_y5.long()
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            x1, x2, x3, x4, x5 = model(inputs)
            # calculate loss
            loss1 = criterion1(x1, target_y1)
            loss2 = criterion2(x2, target_y2)
            loss3 = criterion3(x3, target_y3)
            loss4 = criterion4(x4, target_y4)
            loss5 = criterion5(x5, target_y5)
            total_loss = loss1 + loss2 + loss3 + loss4 + loss5
            # credit assignment
            total_loss.backward()
            print("epoch: {}, batch: {}, total loss: {}, loss1: {}, loss2: {}, loss3: {}, loss4: {}, loss5: {}"
                  .format(epoch, i, total_loss.data, loss1.data, loss2.data, loss3.data, loss4.data, loss5.data))
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
    pred1_list, true1_list = [], []
    pred2_list, true2_list = [], []
    pred3_list, true3_list = [], []
    pred4_list, true4_list = [], []
    pred5_list, true5_list = [], []
    for i, (inputs, target_y1, target_y2, target_y3, target_y4, target_y5) in enumerate(test_dl):
        # evaluate the model on the test set
        pred1, pred2, pred3, pred4, pred5 = model(inputs)
        # retrieve numpy array
        pred1 = pred1.detach().numpy()
        pred2 = pred2.detach().numpy()
        pred3 = pred3.detach().numpy()
        pred4 = pred4.detach().numpy()
        pred5 = pred5.detach().numpy()
        true1 = target_y1.numpy().reshape((len(target_y1), 1))
        true2 = target_y2.numpy().reshape((len(target_y2), 1))
        true3 = target_y3.numpy().reshape((len(target_y3), 1))
        true4 = target_y4.numpy().reshape((len(target_y4), 1))
        true5 = target_y5.numpy().reshape((len(target_y5), 1))
        # convert to class labels
        pred1 = argmax(pred1, axis=1).reshape((len(pred1), 1))
        pred2 = argmax(pred2, axis=1).reshape((len(pred2), 1))
        pred3 = argmax(pred3, axis=1).reshape((len(pred3), 1))
        pred4 = argmax(pred4, axis=1).reshape((len(pred4), 1))
        pred5 = argmax(pred5, axis=1).reshape((len(pred5), 1))
        # store
        pred1_list.append(pred1)
        pred2_list.append(pred2)
        pred3_list.append(pred3)
        pred4_list.append(pred4)
        pred5_list.append(pred5)
        true1_list.append(true1)
        true2_list.append(true2)
        true3_list.append(true3)
        true4_list.append(true4)
        true5_list.append(true5)

    # calculate accuracy
    predictions, actuals = vstack(pred1_list), vstack(true1_list)
    acc1 = accuracy_score(actuals, predictions)
    predictions, actuals = vstack(pred2_list), vstack(true2_list)
    acc2 = accuracy_score(actuals, predictions)
    predictions, actuals = vstack(pred3_list), vstack(true3_list)
    acc3 = accuracy_score(actuals, predictions)
    predictions, actuals = vstack(pred4_list), vstack(true4_list)
    acc4 = accuracy_score(actuals, predictions)
    predictions, actuals = vstack(pred5_list), vstack(true5_list)
    acc5 = accuracy_score(actuals, predictions)

    return acc1, acc2, acc3, acc4, acc5


if __name__ == '__main__':
    # prepare the data
    train_dl, test_dl = prepare_data()
    print(len(train_dl.dataset), len(test_dl.dataset))
    # have a look at train data
    print(train_dl.dataset[0])
    # create CNN model
    model = CNNModel()
    print(model)
    # train the model
    train_model(train_dl, model)
    torch.save(model.state_dict(), 'captcha_recognize.pth')
    # evaluate the model
    acc1, acc2, acc3, acc4, acc5 = evaluate_model(test_dl, model)
    print('Accuracy 1: %.4f, Accuracy 2: %.4f, Accuracy 3: %.4f, Accuracy 4: %.4f, Accuracy 5: %.4f' \
          % (acc1, acc2, acc3, acc4, acc5))
