# -*- coding: utf-8 -*-
# @Time : 2021/1/29 22:48
# @Author : Jclian91
# @File : model_train.py
# @Place : Yangpu, Shanghai
import json
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from params import ModelConfig, ModelTokenizer, ModelForTokenClassification

from params import dataset, train_file_path, test_file_path
from params import MAX_SEQ_LENGTH, BASE_MODEL_DIR, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, EPOCHS, LEARNING_RATE
from load_data import read_data

# tokenizer and label_2_id_dict
with open("{}_label2id.json".format(dataset), "r", encoding="utf-8") as f:
    tag2idx = json.loads(f.read())
    idx2tag = {v: k for k, v in tag2idx.items()}


class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            # pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = [0]+self.labels[index]
        label.extend([0] * MAX_SEQ_LENGTH)
        label = label[:MAX_SEQ_LENGTH]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len


# Creating the customized model
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        config = ModelConfig.from_pretrained("../{}".format(BASE_MODEL_DIR), num_labels=len(list(tag2idx.keys())))
        self.l1 = ModelForTokenClassification.from_pretrained('../{}'.format(BASE_MODEL_DIR), config=config)
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 200)

    def forward(self, ids, mask, labels):
        output_1 = self.l1(ids, mask, labels=labels)
        # output_2 = self.l2(output_1[0])
        # output = self.l3(output_2)
        return output_1


def flat_accuracy(preds, labels):
    flat_preds = np.argmax(preds, axis=2).flatten()
    flat_labels = labels.flatten()
    return np.sum(flat_preds == flat_labels)/len(flat_labels)


def valid(model, testing_loader):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))


if __name__ == '__main__':

    # Preparing for CPU or GPU usage
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = ModelTokenizer.from_pretrained('../{}'.format(BASE_MODEL_DIR))

    # Creating the Dataset and DataLoader for the neural network
    train_sentences, train_labels = read_data(train_file_path)
    train_labels = [[tag2idx.get(l) for l in lab] for lab in train_labels]
    test_sentences, test_labels = read_data(test_file_path)
    test_labels = [[tag2idx.get(l) for l in lab] for lab in test_labels]
    print("TRAIN Dataset: {}".format(len(train_sentences)))
    print("TEST Dataset: {}".format(len(test_sentences)))

    training_set = CustomDataset(tokenizer, train_sentences, train_labels, MAX_SEQ_LENGTH)
    testing_set = CustomDataset(tokenizer, test_sentences, test_labels, MAX_SEQ_LENGTH)

    train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # train the model
    model = BERTClass()
    model.to(dev)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        model.train()
        for _, data in enumerate(training_loader):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)

            loss = model(ids, mask, labels=targets)[0]

            # optimizer.zero_grad()
            if _ % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {_}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # model evaluate
    valid(model, testing_loader)
    torch.save(model.state_dict(), '{}_{}_ner.pth'.format(BASE_MODEL_DIR, dataset))