# chatbot
This chatbot uses a simple feed-forward network made in PyTorch in order to understand and process data. 

# Training
All the training data comes from the intents.json file which contains different sets of intentions (you can also add your own functions to make the bot even better). data.py processes the intentions and tokenizes them into seperate words. All the words which the bot can understand is stored in a list called allWords. The input data is a list of 1's and 0's with the same length as allWords. If the input string contains a word that's also in the allWords list a 1 is inserted in the same position in the input tensor. 

# Input data example
allWords = ["hi", "do", "day", "you", "how", "good", "know", "are"].
If the user input string = "good day" the input tensor will look like following: [0, 0, 1, 0, 0, 1, 0, 0]. 

# Before you try to run
Make sure to have the following libraries installed: pytorch, flask and numpy.
First run the train.py file to train the network and to save the parameters. Then you can run app.py to start the chatbot.
