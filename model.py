import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

class MyClassifier(nn.Module):
    def __init__(self, config):
        super(MyClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.lstm = nn.LSTM(config.d_embed, config.d_hidden)
        self.avgpool = nn.AvgPool1d(config.d_hidden)
        self.hidden2tag = nn.Linear(config.d_hidden, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        batch_size = self.config.batch_size
        return (autograd.Variable(torch.zeros(1, batch_size, self.config.d_hidden)),
                autograd.Variable(torch.zeros(1, batch_size, self.config.d_hidden)))

    def forward(self, content):
        embeds = self.embed(content)
        #print("\nlen(embeds):", len(embeds))
        lstm_out, hidden = self.lstm(embeds.view(len(content), 1, -1), self.hidden)
        tag = self.avgpool(lstm_out)
        tag2tag = nn.Linear(len(tag), 1)
        tag = tag2tag(tag[:, 0, 0])
        return tag


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        # kernal_size = (K,D)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False

    def forward(self, inputs, is_training=False):
        inputs = self.embedding(inputs).unsqueeze(1)  # (B,1,T,D)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)

        concated = torch.cat(inputs, 1)

        if is_training:
            concated = self.dropout(concated)  # (N,len(Ks)*Co)
        out = self.fc(concated)
        return F.log_softmax(out, 1)

