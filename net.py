import torch.nn as nn
import torch.nn.functional as F
from data import Data, dir

data = Data(dir)

# Simple feed-forward network. Input and output size is taken from data.py
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(data.inputSize(), 64) # input layer
        self.fc2 = nn.Linear(64, 64) # hidden layer
        self.fc3 = nn.Linear(64, data.outputSize()) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
