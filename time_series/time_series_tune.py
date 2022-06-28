from multiprocessing.spawn import prepare
from re import X
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import random

torch.manual_seed(0)

from sklearn.preprocessing import MinMaxScaler
import pickle

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

def train(config, data_dir, checkpoint_dir=None):

    random.seed(1234)
    np.random.seed(1234)

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

        if epoch % 100 == 0:
            print("Epoch: %d, training loss: %1.5f, validation loss: %1.5f" % (epoch, loss.item(), val_loss.item()))

        tune.report(loss = val_loss.item())

    lstm.eval()

def main(tag, num_samples=100, max_num_epochs=100):

    random.seed(1234)
    np.random.seed(1234)
    
    data_dir = os.path.abspath("./data")

    load_data(tag, data_dir)

    config = {
        'seq_length' : tune.choice([6, 8, 10]),
        'train_proportion' : tune.choice([0.88, 0.90, 0.92]),
        'num_layers' : tune.choice([1]),
        'num_classes' : tune.choice([1]),
        'input_size' : tune.choice([1]),
        'hidden_size' : tune.choice([2]),
        'learning_rate' : tune.choice([0.01]),
        'num_epochs' : tune.choice([3000]),
        "seed": 1234,
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"])

    result = tune.run(
        tune.with_parameters(train, data_dir = data_dir),
        resources_per_trial={"cpu": 2},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    # best_trained_model = LSTM(
    #     best_trial.config['num_classes'],
    #     best_trial.config['num_layers'],
    #     best_trial.config['input_size'],
    #     best_trial.config['hidden_size'],
    #     best_trial.config['seq_length'],
    # )

if __name__ == '__main__':  

    tag = 'android'
    main(tag)