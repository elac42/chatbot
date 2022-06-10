import json
import random

dir = "data/intents.json"

class Data:
    def __init__(self, jsonfile):
        self.jsonfile = jsonfile
        self.prepareData()

    # Function to tokenize string into words. Returns list of strings
    def tokenize(self, pattern):
        words = []
        pattern = pattern.lower()
        word = ""
        for i in range(len(pattern)):
            if pattern[i] == ' ' or pattern[i] == "!" or pattern[i] == "?" or pattern[i] == ',':
                if len(word) != 0:
                    words.append(word)
                word=""

            else:
                word+=pattern[i]

        if len(word) != 0:
            words.append(word)

        return words

    # Function that creates a new list without duplicates. Used on the allWords list. 
    def checkDuplicate(self, pattern):
        newList = []
        for word in pattern:
            if word not in newList:
                newList.append(word)

        return newList

    def prepareData(self):
        # Open file in read-mode
        f = open(self.jsonfile, 'r')
        data = json.load(f)
        self.allWords = []
        self.categories = []
        for intent in data['intents']:
            category = intent['tag']
            self.categories.append(category)
            for pattern in intent['patterns']:
                # Extend allWords[] with tokenized string
                self.allWords.extend(self.checkDuplicate(self.tokenize(pattern)))

    # Returns list of 1's and 0's (input data) instead of words
    def convertData(self, command):
        data = []
        tokenizedCommand = self.tokenize(command)
        for i in range(len(tokenizedCommand)):
            sequence = [0.0 for i in range(len(self.allWords))]
            for j in range(len(self.allWords)):
                # set data[j] to 1 if it contains allWords[i]
                if tokenizedCommand[i] == self.allWords[j]:
                    sequence[j] = 1.0
                    break
            data.append(sequence)

        return data

    # Returns two lists. One for input data (x_train) and one for desired output (y_train)
    def trainingData(self):
        f = open(self.jsonfile, 'r')
        data = json.load(f)
        duplicate = False
        x_train = []
        y_train = []

        for idx, intent in enumerate(data['intents']):
            category = intent['tag']
            for pattern in intent['patterns']:
                # Add index of category to y_train
                y_train.append(idx)
                x_train.append(self.convertData(pattern.lower()))

        return x_train, y_train

    def getResponse(self, idx):
        f = open(self.jsonfile, 'r')
        data = json.load(f)
        # Randomize response
        response = data['intents'][idx]['responses'][random.randint(0, len(data['intents'][idx]['responses'])-1)]

        return response

    # Returns amount of words that the bot can recognize / input size for neural network.
    def inputSize(self):
        return len(self.allWords)

    # Returns amount of categories / output size for neural network.
    def outputSize(self):
        return len(self.categories)
