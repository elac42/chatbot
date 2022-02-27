from flask import Flask, render_template, request, jsonify
from data import Data, dir
import numpy as np
from train import savePath
import torch
from net import Net

data = Data(dir)

if torch.cuda.is_available() == True:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# Load pre-trained Model
params = torch.load(savePath)
net = Net().to(device)
net.load_state_dict(params)
net.eval()

def chat(command):
    x = torch.from_numpy(np.array(data.convertData(command), dtype=np.float32)).to(device)
    out = net(x)
    # Returns the index of the maximum value of the output layer
    _, predict = torch.max(out, dim=0)
    response = data.getResponse(int(predict.item()))

    return response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    botMsg = request.form['response']
    if botMsg:
        botMsg = chat(botMsg)
        return jsonify({"response": botMsg})
    return({"response": "fail"})


if __name__ == '__main__':
    app.run(debug=True)
