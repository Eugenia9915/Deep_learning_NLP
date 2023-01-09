import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader #, random_split

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizerFast
# from transformers import Trainer, TrainingArguments
from transformers import DistilBertForSequenceClassification

# warning setting
pd.options.mode.chained_assignment = None

# read train/test data
train_pd = pd.read_csv('./train.csv')
test_pd = pd.read_csv('./test.csv')

train_pd = train_pd.drop(['ID'], axis=1)
x, y = train_pd['review'], train_pd['sentiment']

# preprocessing
for i in range(len(x)):
    data = x[i]
    data = data.replace('<br />','')
    data = data.replace(' -',' ')
    data = data.replace('"','')
    data = data.replace('(','')
    data = data.replace(')','')
    data = data.replace('*','')
    data = data.replace(" '",'')
    data = data.replace("' ",'')
    data = re.sub(r"https?://[A-Za-z0-9./]+", ' ', data)
    data = re.sub(r" +", ' ', data)
    x[i] = data

# split training set and validation set
x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state = 777, train_size=0.9)

x_train = x_train.to_list()
y_train = y_train.to_list()
x_valid = x_valid.to_list()
y_valid = y_valid.to_list()

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# get all vocabulary
# tokenizer.vocab.keys()

class myDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
print(type(x),type(y))

# padding可以改看看
inputs1 = tokenizer(x_train, padding='max_length', truncation=True)
inputs2 = tokenizer(x_valid, padding='max_length', truncation=True)

train_d = myDataset(inputs1, y_train)
valid_d = myDataset(inputs2, y_valid)


m = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
m.load_state_dict(torch.load("best.npy"))
m.to(device)

def to_check_result(test_encoding):
  input_ids = torch.tensor(test_encoding['input_ids']).to(device)
  attention_mask = torch.tensor(test_encoding['attention_mask']).to(device)
  with torch.no_grad():
    outputs = m(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
  y = np.argmax(outputs[0].to('cpu').numpy())
  return y

l2 = []
for i in test_pd['review']:
  test_encoding1 = tokenizer(i, truncation=True, padding=True)
  op = to_check_result(test_encoding1)
  l2.append(op)

result_dict = {'ID': test_pd['ID'], 'sentiment': l2}
pd.DataFrame(result_dict).to_csv('result.csv', columns=['ID','sentiment'])

