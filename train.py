import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import long, optim
import numpy as np
from data import Data, dir
from net import Net
from sklearn.utils import shuffle

# Path to save network
savePath = "model.pth"

# Function to create batches
def generateBatch(x_train, y_train, batch_size=1, shuffle_data=False):
    # Shuffle training data in case shuffle=true.
    if shuffle_data:
        x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_data = []
    y_data = []
    # Start index
    idx = 0
    # Get amount of samples
    amount = len(x_train)
    while amount > 0:
        x_sublist = []
        y_sublist = []
        if amount >= batch_size:
            for i in range(batch_size):
                x_sublist.append(x_train[i+idx])
                y_sublist.append(y_train[i+idx])
            amount-=batch_size
            idx+=batch_size
        else:
            for i in range(amount):
                x_sublist.append(x_train[i+idx])
                y_sublist.append(y_train[i+idx])
            amount = 0
        x_data.append(x_sublist)        
        y_data.append(y_sublist)


    return x_data, y_data

if __name__ == '__main__':
    data = Data(dir)

    # Get training data
    x_train, y_train = data.trainingData()
    # Convert x_train and y_train to numpy arrays. Will be needed to convert to tensors later.
    x_train = np.array(x_train, dtype=object)
    y_train = np.array(y_train, dtype=np.compat.long)
    # If cuda gpu is available use gpu (faster computation). Else use cpu.
    deviceType = ""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        deviceType = "gpu"
    else:
        device = torch.device("cpu")
        deviceType = "cpu"
    print("Training on", deviceType)

    # Input size (sequence length).
    input_size = data.inputSize()
    # Amount of neurons in the hidden state. 
    hidden_size = 256
    # Number of output classes.
    num_classes = data.outputSize()
    # Call on Net() and run it on device.
    net = Net(input_size, hidden_size, num_classes).to(device)

    # Loss function. Since this is a classification problem we use CrossEntropy Loss.
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Epochs. This is the amount of times that the network will see ALL data.
    epochs = 10

    print("Training network with", epochs, "epochs")
    for epoch in range(epochs):
        totalLoss = 0
        idx = 0
        #x_batch, y_batch = generateBatch(x_train, y_train, batch_size=8, shuffle_data=True)
        for i in range(len(x_train)):
            hidden = torch.zeros(hidden_size).to(device)
            x = x_train[i]
            int(len(x))
            y = y_train[i]
            x = torch.from_numpy(np.array(x, dtype=np.float32))
            y = torch.from_numpy(np.array(y))
            y = y.to(long)
            x = x.to(device)
            y = y.to(device)
            for i in range(len(x)):
                output, hidden = net(x[i], hidden)
            loss = criterion(output, y)
            # Reset gradients (change to weights) from the last update
            optimizer.zero_grad()
            # Backpropagate loss. Used to determine the change that's needed to the weights
            loss.backward()
            # Make changes to weights
            optimizer.step()
            totalLoss+=loss.item()
            idx+=1
        print(epoch+1, "loss:", totalLoss/idx)

    # Save the current network
    try:
        torch.save(net.state_dict(), savePath)
        print("Model succesfully saved in", savePath)
    except:
        print("Failed to save model at", savePath)
