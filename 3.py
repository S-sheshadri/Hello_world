import csv
import string
headline_list=[]
with open('Hello_world/news.csv') as csv_file:
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
#tokenizer = get_tokenizer("basic_english")
tokenizer = get_tokenizer("spacy")
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
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
# input_size = 784 # 28x28
num_classes = total_words
num_epochs = 2
batch_size = 1
learning_rate = 0.001

input_size = 1
sequence_length = max_sequence_len
hidden_size = 128
num_layers = 2

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        self.dropout = nn.Dropout(0.25) 
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        #out, _ = self.rnn(x, h0)  
        # or:
        out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
        out = self.dropout(out)
        out = self.fc(out)
        # out: (n, 10)
        return out


model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    i = 0
    for predictor in padded_predictors:
        input_predictor = predictor.unsqueeze(0)
        input_predictor = input_predictor.unsqueeze(-1)
        input_predictor = input_predictor.float()
        target = torch.LongTensor([labels[i]])
        i = i+1
        output = model(input_predictor)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print("Epoch"+str(epoch)+"\ni="+str(i)+"\nLoss"+str(loss))

import random
for i in range(5):
    max_length = 20
    random_predictor = random.choice(ngrams)

    random_padded = [torch.tensor(random_predictor), predictor]

    random_padded = pad_sequence(random_padded, batch_first=True)

    input_random = random_padded[0]
    input_random = input_random.unsqueeze(0)
    input_random = input_random.unsqueeze(-1)
    input_random = input_random.float()
    output =  model(input_random)

    print("INPUT:")
    for i in random_predictor:
        print(ind2word[i])
    topv, topi = output.topk(1)
    topi = topi[0][0]
    print("PREDICTED:")
    print(ind2word[topi.item()])
max_length = 5
def sample():
    i = 0
    random_predictor = random.choice(ngrams)
    headline = ""
    for word in random_predictor:
        headline += " "+ind2word[word]
    while(len(random_predictor)<max_sequence_len):
        random_padded = [torch.tensor(random_predictor), predictor]

        random_padded = pad_sequence(random_padded, batch_first=True)

        input_random = random_padded[0]
        input_random = input_random.unsqueeze(0)
        input_random = input_random.unsqueeze(-1)
        input_random = input_random.float()
        output =  model(input_random)

        topv, topi = output.topk(1)
        topi = topi[0][0]

        word = ind2word[topi.item()]
        headline += " "+word
        random_predictor.append(topi.item())
    return headline

    


























