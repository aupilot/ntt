import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN1(nn.Module):
    def __init__(self, input_depth, dropout=0.25):

        super(CNN1, self).__init__()
        self.name = 'cnn1 simple'

        self.dropout = dropout

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_depth, 64, kernel_size=17),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, kernel_size=7, dilation=3),
            nn.LeakyReLU(),
            nn.AvgPool1d(5, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=8, dilation=4),
            nn.LeakyReLU(),
            nn.AvgPool1d(7, stride=4))

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=8, dilation=4),
            nn.LeakyReLU(),
            nn.AvgPool1d(256))

        self.flatten = Flatten()
        self.fc = nn.Linear(256, 6)
        self.lsm = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(out)
        out = F.dropout(out, p=0.25, training=self.training)
        out = self.fc(out)
        out = self.lsm(out)

        return out
