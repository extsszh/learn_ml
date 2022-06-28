from multiprocessing.spawn import prepare
from re import X
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

import os

from sklearn.preprocessing import MinMaxScaler
import pickle

torch.manual_seed(0)

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def prepare_data_sets(training_set, seq_length, train_proportion, data_dir = './data'):

    sc = MinMaxScaler()

    training_data = sc.fit_transform(training_set)

    with open(data_dir + '/sc', 'wb') as f:
        pickle.dump(sc, f)

    x, y = sliding_windows(training_data, seq_length)

    train_size = int(len(y) * train_proportion)

    x = Variable(torch.Tensor(np.array(x)))
    y = Variable(torch.Tensor(np.array(y)))

    train_x = Variable(torch.Tensor(np.array(x[0:train_size])))
    train_y = Variable(torch.Tensor(np.array(y[0:train_size])))

    test_x = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    test_y = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    return (x, y, train_x, train_y, test_x, test_y)

def load_data(tag, data_dir = './data'):
    
    with open('time_series_quarterly.pickle', 'rb') as f:
        data = pickle.load(f)

    training_set = pd.DataFrame(data).loc[:, [tag]].values

    with open(data_dir + '/training_set', 'wb') as f:
        pickle.dump(training_set, f)

    return training_set

class LSTM(nn.Module):

    def __init__(self, num_classes, num_layers, input_size, hidden_size, seq_length):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

def train(config, data_dir):

    lstm = LSTM(
        num_classes = config['num_classes'],
        num_layers = config['num_layers'],
        input_size = config['input_size'],
        hidden_size = config['hidden_size'],
        seq_length = config['seq_length']
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=config['learning_rate'])

    with open(data_dir + '/training_set', 'rb') as f:
        training_set = pickle.load(f)

    x, y, train_x, train_y, test_x, test_y = prepare_data_sets(training_set, config['seq_length'], config['train_proportion'])

    with open(data_dir + '/x', 'wb') as f:
        pickle.dump(x, f)
    with open(data_dir + '/y', 'wb') as f:
        pickle.dump(y, f)
    with open(data_dir + '/train_x', 'wb') as f:
        pickle.dump(train_x, f)
    with open(data_dir + '/train_y', 'wb') as f:
        pickle.dump(train_y, f)
    with open(data_dir + '/test_x', 'wb') as f:
        pickle.dump(test_x, f)
    with open(data_dir + '/test_y', 'wb') as f:
        pickle.dump(test_y, f)

    for epoch in range(config['num_epochs']):

        outputs = lstm(train_x)
        optimizer.zero_grad()
        
        loss = criterion(outputs, train_y)
    
        loss.backward()
        
        optimizer.step()

        lstm.eval()

        with torch.no_grad():
            val_outputs = lstm(test_x)
            val_loss = criterion(val_outputs, test_y)
        
        lstm.train()

        print("Epoch: %d, training loss: %1.5f, validation loss: %1.5f" % (epoch, loss.item(), val_loss.item()))

    lstm.eval()

    return lstm

def make_projections(model, num_quarters, data_dir = './data'):

    with open(data_dir + '/x', 'rb') as f:
        x = pickle.load(f)

    predict_y = model(x)

    predict_x = torch.clone(x)
    project_y = torch.clone(predict_y)

    predict_x_list = predict_x.tolist()
    project_y_list = project_y.tolist()

    for _ in range(num_quarters):
        nextX = predict_x_list[-1][1:] + [project_y_list[-1]]
        predict_x_list.append(nextX)
        project_y_list.append(model(torch.tensor([nextX]))[0].tolist())

    project_y = torch.tensor(project_y_list)

    return project_y

def plot(tag, project_y, data_dir = './data'):
    
    with open(data_dir + '/sc', 'rb') as f:
        sc = pickle.load(f)

    with open(data_dir + '/y', 'rb') as f:
        y = pickle.load(f)

    with open(data_dir + '/train_y', 'rb') as f:
        train_y = pickle.load(f)

    y_plot = sc.inverse_transform(y.data.numpy())

    project_y_plot = sc.inverse_transform(project_y.data.numpy())

    plt.axvline(x = len(train_y), c='r', linestyle='--')
    plt.axvline(x = len(y), c='b', linestyle='--')

    plt.plot(y_plot)
    plt.plot(project_y_plot)
    plt.suptitle(f'Time Series Projection for [{tag}]')
    plt.show()

def main(tag):

    data_dir = os.path.abspath("./data")

    load_data(tag, data_dir)

    config = {
        'seq_length' : 10,
        'train_proportion' : 0.88,
        'num_layers' : 1,
        'num_classes' : 1,
        'input_size' : 1,
        'hidden_size' : 2,
        'learning_rate' : 0.1,
        'num_epochs' : 1000
    }

    model = train(config, data_dir)

    project_y = make_projections(model, 12, data_dir = './data')

    plot(tag, project_y, data_dir = './data')

if __name__ == '__main__':  

    tag = 'angular'
    main(tag)