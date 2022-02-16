# chatbot
This chatbot uses a simple feed-forward network made in PyTorch in order to understand and process data. 

# Training
All the training data comes from the intents.json file which contains different sets of intentions. Data.py processes the intentions and tokenizes them into seperate words. All the words which the bot can understand is stored in a list called allWords. The input data is a list of 1's and 0's with the same length as allWords. If the input string contains a word that's also in the allWords list a 1 is inserted in the same position in the input tensor. 

# Input data example
allWords = ["hi", "do", "day", "you", "how", "good", "know", "are"].
If the user input string = "good day" the input tensor would look like the following: [0, 0, 1, 0, 0, 1, 0, 0].
