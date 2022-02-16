from data import Data, dir
from flask import Flask
import numpy as np
from train import savePath
import torch
from net import Net

data = Data(dir)

if __name__ == '__main__':
    if torch.cuda.is_available() == True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # Load pre-trained Model
    params = torch.load(savePath)
    net = Net().to(device)
    net.load_state_dict(params)
    net.eval()
    while True:
        command = input("Ask something: ")
        x = torch.from_numpy(np.array(data.convertData(command), dtype=np.float32)).to(device)
        out = net(x)
        # Returns the index of the maximum value of the output layer
        _, predict = torch.max(out, dim=0)
        response = data.getResponse(int(predict.item()))
        print(response)
