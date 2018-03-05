#pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

#other lib
import random
import numpy as np
import re
import pandas as pd
import gc
import jieba
# import matplotlib.pyplot as plt
from copy import deepcopy
import time
#my lib
from model import CNNClassifier

#some define
flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(1024)

USE_CUDA = torch.cuda.is_available()

# gpus = [0]
# torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    #print(idxs)
    return Variable(LongTensor(idxs))

#read from file
reader = pd.read_csv('BDCI2017-360/train.tsv', sep='\t', iterator=True, usecols=range(0, 4))
chunk_size = 500
chunk_total = 2000

start_time = time.time()

#get word to index
word_to_ix = {'<PAD>': 0, '<UNK>': 1}
index = 0
# vocab = set()
while index < chunk_total:
    chunk = reader.get_chunk(chunk_size)
    for i in range(chunk_size):
        # print(i)
        sentencei = chunk.ix[i + index, 2]
        if not isinstance(sentencei, str):
            #print(sentencei)
            continue
        sentencei = re.sub('\d', '#', sentencei)
        #print(sentencei)
        seg_list = set(jieba.cut(sentencei))
        #print(seg_list)
        # vocab = vocab | seg_list
        for word in seg_list:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    print(index)
    index += chunk_size
# print(len(vocab))
#
# for vo in vocab:
#     if word_to_ix.get(vo) is None:
#         word_to_ix[vo] = len(word_to_ix)

print("time elapsed: {:.2f}s".format(time.time() - start_time))

#tag_to_ix
tag_to_ix = {"NEGATIVE": 0, "POSITIVE": 1}

del reader
gc.collect()

#train
chunk_size = 10
def pad_to_batch(batch):
    x,y = zip(*batch)
    max_x = max([s.size(1) for s in x])
    x_p = []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            x_p.append(torch.cat([x[i], Variable(LongTensor([word_to_ix['<PAD>']] * (max_x - x[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(x[i])
    return torch.cat(x_p), torch.cat(y).view(-1)

EPOCH = 5
BATCH_SIZE = 50
KERNEL_SIZES = [3,4,5]
KERNEL_DIM = 75
LR = 0.001

model = CNNClassifier(len(word_to_ix), 150, len(tag_to_ix), KERNEL_DIM, KERNEL_SIZES)
model = nn.DataParallel(model)
if USE_CUDA:
    model = model.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

losses_mean = []
for epoch in range(EPOCH):
    losses = []
    reader = pd.read_csv('BDCI2017-360/train.tsv', sep='\t', iterator=True, usecols=range(0, 4))
    index = 0
    while index < chunk_total:
        chunk = reader.get_chunk(chunk_size)
        X = []
        y = []
        data_error = 0
        for i in range(chunk_size):
            tag = chunk.ix[i + index, 3]

            if tag != 'POSITIVE' and tag != 'NEGATIVE':
                data_error += 1
                #print(tag)
                break
            sentencei = chunk.ix[i + index, 2]
            if not isinstance(sentencei, str):
                #print(sentencei)
                data_error += 1
                break
            sentencei = re.sub('\d', '#', sentencei)
            seg_list = list(jieba.cut(sentencei))
            X.append(seg_list)
            #tag = chunk.ix[i+index, 3]
            y.append(tag)
        if data_error > 0:
            #print(index)
            index += chunk_size
            continue
        X_p, y_p = [], []
        for pair in zip(X, y):
            X_p.append(prepare_sequence(pair[0],word_to_ix).view(1, -1))
            y_p.append(Variable(LongTensor([tag_to_ix[pair[1]]])).view(1, -1))

        #train_data = data_p
        inputs, targets = pad_to_batch(list(zip(X_p, y_p)))
        del X_p, y_p
        gc.collect()

        model.zero_grad()
        preds = model(inputs, True)
        loss = loss_function(preds, targets)
        losses.append(loss.data.tolist()[0])
        loss.backward()
        optimizer.step()

        gc.collect()
        index += chunk_size
        if index % 100 == 0:
            print("[%d/%d/%d] mean_loss : %0.2f" % (epoch, index, chunk_total, np.mean(losses)))
            losses_mean.append(np.mean(losses))
            losses = []

# plt.plot(np.squeeze(losses_mean))
# plt.savefig("train2.jpg") 
accuracy = 0

accuracyes_mean = []
chunk_total *= 1.2
while index < chunk_total:
    accuracyes = []
    accuracy = 0
    chunk = reader.get_chunk(chunk_size)
    X = []
    y = []
    data_error = 0
    for i in range(chunk_size):
        tag = chunk.ix[i + index, 3]

        if tag != 'POSITIVE' and tag != 'NEGATIVE':
            data_error += 1
            #print(tag)
            break
        sentencei = chunk.ix[i + index, 2]
        if not isinstance(sentencei, str):
            #print(sentencei)
            data_error += 1
            break
        sentencei = re.sub('\d', '#', sentencei)
        seg_list = list(jieba.cut(sentencei))
        X.append(seg_list)
        #tag = chunk.ix[i+index, 3]
        y.append(tag)
    if data_error > 0:
        #print(index)
        index += chunk_size
        continue
    X_p, y_p = [], []
    for pair in zip(X, y):
        X_p.append(prepare_sequence(pair[0],word_to_ix).view(1, -1))
        y_p.append(Variable(LongTensor([tag_to_ix[pair[1]]])).view(1, -1))

    test_data = list(zip(X_p, y_p))

    #inputs, targets = pad_to_batch(list(zip(X_p, y_p)))
    del X_p, y_p
    gc.collect()

    for test in test_data:
        pred = model(test[0]).max(1)[1]
        pred = pred.data.tolist()[0]
        target = test[1].data.tolist()[0][0]
        if pred == target:
            accuracy += 1

    #print(accuracy / len(test_data) * 100)
    accuracyes.append(accuracy / len(test_data) * 100)
    index += chunk_size
    if index % 100 == 0:
        print("[%d/%d] mean_accuarcy : %0.2f" % (index, chunk_total, np.mean(accuracyes)))
        accuracyes_mean.append(np.mean(accuracyes))
        accuracyes = []

print("total accuarcy is %0.2f" % (np.mean(accuracyes_mean)))










