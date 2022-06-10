# Training
All the training data comes from the intents.json file which contains different sets of intentions (you can also add your own intents to make the bot even better). data.py processes the intentions and tokenizes them into seperate words. All the words which the bot can understand is stored in a list called allWords. The input data is a list containing one-hot encoded vectors of the words. 

# Input data example
allWords = ["hi", "do", "day", "you", "how", "good", "know", "are"].
If the user input string = "good day" the input tensor will look like following: [[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]]. Notice how the vector representing the word "good" comes before the vector representing the word "day" since we want to process the data in sequences. We first send the vector representing the word "good" and save that output as a "hidden state". We then combine the hidden state representing the word "good" and combines with the next word, "day". When we process the combined input we will get the resulting output which represents the words "good" and "day" in that specific order and can now predict the wanted output.   

# Before you try to run
Make sure to have the following libraries installed: pytorch, flask and numpy.
First run the train.py file to train the network and to save the parameters. Then you can run app.py to start the chatbot.
