import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data import Data, dir
from net import Net

# Path to save network
savePath = "model.pth"

class dataset(Dataset):
    def __init__(self):
        self.samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.samples

if __name__ == '__main__':
    data = Data(dir)

    # Get training data
    x_train, y_train = data.trainingData()

    # Convert x_train and y_train to numpy arrays. Will be needed to convert to tensors later.
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.long)
    # If cuda gpu is available use gpu (faster computation). Else use cpu.
    deviceType = ""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        deviceType = "gpu"
    else:
        device = torch.device("cpu")
        deviceType = "cpu"
    print("Training on", deviceType)

    # Call on Net() and run it on device.
    net = Net().to(device)

    # Batch size. The number of samples shown to the network every time before making changes to weights
    batch_size = 8
    datas = dataset()
    loader = DataLoader(dataset=datas, batch_size=batch_size, shuffle=True)

    # Loss function. Since this is a classification problem we use CrossEntropy Loss.
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    # Epochs. This is the amount of times that the network will see ALL data.
    epochs = 10

    print("Training network with", epochs, "epochs")
    for epoch in range(epochs):
        totalLoss = 0
        idx = 0
        for (x_data, y_data) in loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            output = net(x_data)
            loss = criterion(output, y_data)
            # Reset gradients (change to weights) from the last update
            optimizer.zero_grad()
            # Backpropagate loss. Used to determine the change that's needed to the weights
            loss.backward()
            # Make changes to weights
            optimizer.step()
            totalLoss+=loss
            idx+=1
        print(epoch+1, "loss:", totalLoss.item())

    # Save the current network
    try:
        torch.save(net.state_dict(), savePath)
        print("Model succesfully saved in", savePath)
    except:
        print("Failed to save model at", savePath)
