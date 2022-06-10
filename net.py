import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple feed-forward network. Input and output size is taken from data.py
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.fcHidden = nn.Linear(input_size+hidden_size, hidden_size)
        self.fcOut = nn.Linear(hidden_size, num_classes) # output layer

    def forward(self, x, hidden):
        # Combine the word at current timestep with the previous words (hidden tensor). Equivalent to python's extend().
        combined = torch.cat((x, hidden), -1)
        hidden = self.fcHidden(combined)
        out = self.fcOut(hidden)

        # Since we're using cross entropy loss we don't need to apply softmax to the outputs.
        # The hidden output will be fed to the network again and be combined with the next word. 
        # If the last word has been combined and fed into the network (fcOut) we finally have our real output vector and we're ready to predict. 
        return out, hidden
