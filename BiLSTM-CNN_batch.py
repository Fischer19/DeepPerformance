# coding: utf-8

# naive LSTM model trained with smooth data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from random import shuffle
import random
import numpy as np
from torch.autograd import Variable
import argparse
import time
import math
import pickle

class DecoderRNN(nn.Module):
    def __init__(self, input_size, augmented_size, hidden_size, output_size, dropout_p = 0):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.augmented_size = augmented_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.verbose = (self.dropout_p != 0)

        self.cnn1 = nn.Conv2d(self.input_size, self.augmented_size, kernel_size=(9,128), padding=(4,0))
        self.relu1 = nn.ReLU()
        #self.cnn2 = nn.Conv2d(self.input_size, self.augmented_size, kernel_size=(3,128), padding=(1,0))
        #self.relu2 = nn.ReLU()
        self.lstm_1 = nn.LSTM(self.augmented_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_3 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_4 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_5 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_6 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_7 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.dropout4 = nn.Dropout(self.dropout_p)
        self.dropout5 = nn.Dropout(self.dropout_p)
        self.dropout6 = nn.Dropout(self.dropout_p)
        # map the output of LSTM to the output space
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        self.batch_size = input.shape[0]
        
        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()
        self.hidden3 = self.init_hidden()
        self.hidden4 = self.init_hidden()
        self.hidden5 = self.init_hidden()
        self.hidden6 = self.init_hidden()
        self.hidden7 = self.init_hidden()

        output = self.cnn1(input.view(self.batch_size,self.input_size, -1,128))
        output = self.relu1(output)
        #print("cnn1 output shape:", output.shape)

        output, self.hidden1 = self.lstm_1(output.view(-1,1,self.augmented_size), self.hidden1)
        if self.verbose:
            output = self.dropout1(output)
        output_1 = output

        output, self.hidden2 = self.lstm_2(output, self.hidden2)
        if self.verbose:
            output = self.dropout2(output)
        output_2 = output

        output, self.hidden3 = self.lstm_3(output + output_1, self.hidden3)  # skip_connection 1
        if self.verbose:
            output = self.dropout3(output)
        output_3 = output

        output, self.hidden4 = self.lstm_4(output + output_2, self.hidden4)  # skip_connection 2
        if self.verbose:
            output = self.dropout4(output)
        output_4 = output

        output, self.hidden5 = self.lstm_5(output + output_3, self.hidden5)  # skip_connection 3
        
        output = self.out(output).view(self.batch_size, -1,self.output_size)
        return output
        
        if self.verbose:
            output = self.dropout5(output)
        output_5 = output

        output, self.hidden6 = self.lstm_6(output + output_4, self.hidden6)  # skip_connection 4
        if self.verbose:
            output = self.dropout6(output)
        output, self.hidden7 = self.lstm_7(output + output_5, self.hidden7)  # skip_connection 5
        
        output = self.out(output).view(self.batch_size, -1,self.output_size)
        # output = self.softmax(output)
        return output

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size // 2, device=device),
                torch.randn(2, 1, self.hidden_size // 2, device=device))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def compute_beat(train_Y, smooth = True, tempo_curve = False):
    Y_beat = []
    Y_tempo = []
    for i in range(len(train_Y)):
        cumulative_beat = 0
        beat = []
        tempo = []
        for j, item in enumerate(train_Y[i]):
            cumulative_beat += item[1] * 0.05
            tempo.append(item[1] * 60)
            beat.append(cumulative_beat)
        Y_beat.append(beat)
        Y_tempo.append(tempo)
    train_Y = Y_beat
    output = []
    if tempo_curve:
        return Y_tempo
    if smooth:
        for i in range(len(train_Y)):
            n = len(train_Y[i])
            smooth_label = np.zeros_like(train_Y[i])
            reversed_smooth_label = np.zeros_like(train_Y[i])
            denominator = 1
            for j, item in enumerate(train_Y[i]):
                denominator += 1
                if j == 0:
                    smooth_label[j] = 1
                    denominator = 1
                else:
                    if np.int(item) - np.int(train_Y[i][j - 1]) == 1:
                        smooth_label[j] = 1
                        denominator = 1
                    else:
                        smooth_label[j] = 1/denominator

            reversed_Y = train_Y[i][::-1]
            for j, item in enumerate(reversed_Y):
                denominator += 1
                if j == n-1:
                    reversed_smooth_label[j] = 1
                    denominator = 1
                else:
                    if np.int(item) - np.int(reversed_Y[j+1]) == 1:
                        reversed_smooth_label[j] = 1
                        denominator = 1
                    else:
                        reversed_smooth_label[j] = 1/denominator
            reversed_smooth_label = reversed_smooth_label[::-1]
            for i in range(len(smooth_label)):
                if reversed_smooth_label[i] > smooth_label[i]:
                    smooth_label[i] = reversed_smooth_label[i]
            output.append(smooth_label)
    return output
#compute_beat(train_Y)

def process_target(train_Y, binary = True, smooth = False):
    output = []
    if binary:
        for i in range(len(train_Y)):
            binary_label = np.zeros_like(train_Y[i][:,0])
            for j, item in enumerate(train_Y[i]):
                if j == 0:
                    binary_label[j] = 1
                else:
                    if np.int(item[0]) - np.int(train_Y[i][j - 1][0]) == 1:
                        binary_label[j] = 1
            output.append(binary_label)
    else:
        if smooth:
            for i in range(len(train_Y)):
                n = len(train_Y[i])
                smooth_label = np.zeros_like(train_Y[i][:,0])
                reversed_smooth_label = np.zeros_like(train_Y[i][:,0])
                denominator = 1
                for j, item in enumerate(train_Y[i]):
                    denominator += 1
                    if j == 0:
                        smooth_label[j] = 1
                        denominator = 1
                    else:
                        if np.int(item[0]) - np.int(train_Y[i][j - 1][0]) == 1:
                            smooth_label[j] = 1
                            denominator = 1
                        else:
                            smooth_label[j] = 1/denominator
                            
                reversed_Y = train_Y[i][::-1]
                for j, item in enumerate(reversed_Y):
                    denominator += 1
                    if j == n-1:
                        reversed_smooth_label[j] = 1
                        denominator = 1
                    else:
                        if np.int(item[0]) - np.int(reversed_Y[j+1][0]) == 1:
                            reversed_smooth_label[j] = 1
                            denominator = 1
                        else:
                            reversed_smooth_label[j] = 1/denominator
                reversed_smooth_label = reversed_smooth_label[::-1]
                for i in range(len(smooth_label)):
                    if reversed_smooth_label[i] > smooth_label[i]:
                        smooth_label[i] = reversed_smooth_label[i]
                output.append(smooth_label)
                
        else:
            for i in range(len(train_Y)):
                label = train_Y[i][:,0]
                output.append(label)
    return output

def input_slicing(train_X, train_Y, path):
    slicing_train_X = []
    slicing_train_Y = []
    slicing_train = []
    target_index = []
    target_composers = ["bee-snt", "moz-snt", "cho-bal", "cho-pld", "bac-wtc"]
    for composer in target_composers:
        for i in range(len(train_X)):
            if i in VAL_INDEX:
                continue
            if(composer in path[i]):
                start_loc, end_loc = 0, 400
                while end_loc <= len(train_X[i]):
                    current_input = train_X[i][start_loc:end_loc]
                    current_label = train_Y[i][start_loc:end_loc]
                    start_loc += 50
                    end_loc += 50
                    slicing_train.append((current_input, current_label))
    shuffle(slicing_train)
    for item in slicing_train:
        X, Y = item
        slicing_train_X.append(X)
        slicing_train_Y.append(Y)
    return np.array(slicing_train_X), np.array(slicing_train_Y)

def penalty_loss(criterion, Y, target, penalty = 10):
    loss = 0
    for i in range(len(target)):
        for j in range(len(target[i])):
            #print(Y[i])
            if int(target[i][j]) == 1 or float(target[i][j]) < 1/7:
                loss += penalty * criterion(Y[i,j], target[i,j])
            else:
                loss += criterion(Y[i,j], target[i,j])
    return loss
    
def train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion):
    decoder_optimizer.zero_grad()
    
    loss = 0
    batch_size = input_tensor.shape[0]
    #print(input_tensor.shape)
    decoder_output= decoder(input_tensor)
    
    #loss += penalty_loss(criterion, decoder_output.view(batch_size,-1),target_tensor)
    loss += criterion(decoder_output.view(batch_size,-1),target_tensor)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
    decoder_optimizer.step()
    return loss.item(), decoder_output

def trainEpochs(decoder, n_epochs, print_every=1000, plot_every=100, learning_rate=0.01, total_batch=100, batch_size = 1,
               penalty=(1, 0.5), gamma=0.1):
    start = time.time()

    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=2, gamma=gamma)
    iter = 0
    for epoch in range(1, n_epochs + 1):
        start, end = 0, batch_size
        
        #verbose = (iter % print_every == 0)
        while end <= total_batch:
            iter += 1
            target_tensor = torch.from_numpy(np.array(train_Y[start:end][:])).to(device).float()
            input_tensor = torch.from_numpy(np.array(train_X[start:end][:])).to(device).float()
            #target_tensor = torch.from_numpy(np.array(train_Y[num])).to(device).float()
            #input_tensor = Variable(input_tensor, requires_grad=True)
            #print(input_tensor.shape, target_tensor.shape, decoder)
            #print(decoder_optimizer, criterion)
            loss, decoder_output = train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion)
            print_loss_total += loss
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                #print(decoder_output.view(-1).detach().cpu().numpy())
                #print(target_tensor)
                #print(decoder_optimizer)
                print("loss%i/%i:"%(iter, n_epochs * (total_batch//batch_size)), print_loss_avg)
                print_loss_total = 0
                
                #training_progress = validation(decoder, train_X, train_Y)
                training_progress = (decoder_output.view(batch_size, -1).cpu().detach().numpy(), train_Y[start:end])
                f = open('/home/yixing/Fischer/DeepPerformance/Bi-LSTM-CNN_batch_progress.pkl', "wb")
                pickle.dump(training_progress, f)
                f.close()
                
                
            start += batch_size
            end += batch_size
        scheduler.step()
            
def validation(decoder, train_X, train_Y):
    input_tensor = torch.from_numpy(np.array(train_X[0][:])).to(device).float()
    decoder_output= decoder(input_tensor)
    return (decoder_output.view(-1).cpu().detach().numpy(), train_Y[0])

if __name__ == "__main__":
    
    device = torch.device(3 if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tempo_curve", help="decode tempo curve",
                        action="store_true")
    parser.add_argument("-lr", "--learning_rate", help="learning rate")
    parser.add_argument("-e", "--epoch", help="epoch")
    parser.add_argument("-o", "--overfitting", help="overfitting test",action="store_true")
    parser.add_argument("-b", "--batch_size", help="batch size")
    args = parser.parse_args()
    if args.tempo_curve:
        smooth = False
        tempo_curve = True
    else:
        smooth = True
        tempo_curve = False
    print("Training model with smooth data:", smooth, "tempo curve data:", tempo_curve)

    with open("/home/yixing/realdata.pkl", "rb") as f:
        VAL_INDEX = [1,2,3,9,16]
        dic = pickle.load(f)
        train_X = dic["X"]
        train_Y = dic["y"]
        path = dic["path"]
        smooth = compute_beat(train_Y, smooth = smooth, tempo_curve = tempo_curve)
        train_X, train_Y = input_slicing(train_X, smooth, path)
        print(train_X.shape)
        print(train_Y.shape)
        maximum_target = len(train_Y)
                        
    if args.overfitting:
        total_batch = args.batch_size
    else:
        total_batch = len(train_Y)

    CLIP = 10
    input_size = 3
    augmented_size = 16
    hidden_size = 256
    output_size = 1

    decoder = DecoderRNN(input_size, augmented_size, hidden_size, output_size).to(device)
    print("total_batch", len(train_Y))
    trainEpochs(decoder, int(args.epoch), print_every=5, learning_rate=float(args.learning_rate), total_batch=int(total_batch), batch_size = int(args.batch_size), gamma=0.5)
    torch.save(decoder.state_dict(), '/home/yixing/Fischer/DeepPerformance/Bi-LSTM-CNN_batch1.pt')
