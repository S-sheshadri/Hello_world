import csv
import string
headline_list=[]
with open('news.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        headline = row[5]
        cleaned_headline = "".join(c for c in headline if c not in string.punctuation).lower()
        cleaned_headline = cleaned_headline.encode("utf8").decode("ascii",'ignore')
        headline_list.append(cleaned_headline)
headline_list = headline_list[1:100]
n_headlines = len(headline_list)
print(n_headlines)

import torchtext
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")

token_map = set()

for headline in headline_list:
    for word in headline.split():
        token_map.add(word)

print(len(token_map))
ind2word = dict(enumerate(token_map))
word2ind = {v: k for k, v in ind2word.items()}

total_words = len(ind2word)

def indexify(corpus):
    index_corpus=[]
    for line in corpus:
        index_line=[]
        for word in tokenizer(line):
            index_line.append(word2ind[word])
        index_corpus.append(index_line)
    return index_corpus

def ngramGenerator(index_corpus):
    ngrams = []
    for index_line in index_corpus:
        for N in range(2,len(index_line)+1):
            grams = [index_line[i:i+N] for i in range(len(index_line)-N+1)]
            if grams:ngrams+=grams

    return ngrams


index_corpus = indexify(headline_list)
ngrams = ngramGenerator(index_corpus)
ngrams = [x for x in ngrams if x and len(x)>1]
import itertools
ngrams.sort()
list(ngrams for ngrams,_ in itertools.groupby(ngrams))
max_sequence_len = max([len(x) for x in ngrams])

from torch.nn.utils.rnn import pad_sequence
import torch
predictors = [ngram[:-1] for ngram in ngrams]
labels = [ngram[-1] for ngram in ngrams]
tensor_predictors = [torch.tensor(predictor) for predictor in predictors]
padded_predictors = pad_sequence(tensor_predictors, batch_first=True)

import numpy as np
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
categorical_labels = [to_categorical(label,total_words) for label in labels]

import torch
import torch.nn.functional as fun 
import torch.nn as nn

"""
input_dim = 2 
hidden_dim = 10
n_layers = 1

lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
batch_size = len(ngrams) 
seq_len = 1
hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
cell_state = torch.randn(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)
inp = torch.tensor(padded_predictors)

out, hidden = lstm_layer(inp, hidden)

"""
from torch.utils.data import TensorDataset, DataLoader
train_data = TensorDataset(torch.tensor(padded_predictors), torch.tensor(categorical_labels))
batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

vocab_size = len(word2ind) 
output_size = len(word2ind)
embedding_dim = 10
hidden_dim = 128
n_layers = 2 


device = torch.device("cpu")

model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.to(device)

lr=0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2
counter = 0
print_every = 10
clip = 5
valid_loss_min = np.Inf

model.train()
for i in range(epochs):
    h = model.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if counter%print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)







