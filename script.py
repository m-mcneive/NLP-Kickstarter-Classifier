import numpy as np 
import csv
from sklearn.utils import shuffle
from keras import models
from keras import layers
from keras import optimizers
from numpy.lib.npyio import savetxt
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import SimpleRNN

file = open('df_text_eng.csv', 'r')
file = file.readlines()


"""
Creates vocabulary from training data. Will condense the vocabulary to a user-specified minimum number of 
occurances for each word. A non-condensed vocabulary would be far too large given the number of training
data samples. 
"""
def generateVocabulary(minOccurances):
    vocabulary = {}
    for line in file:
        line = line.split()
        for word in line[1:-1]:
            if word in vocabulary.keys():
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1

    vocabulary_refined = []
    for word in vocabulary.keys():
        if vocabulary[word] > minOccurances:
            vocabulary_refined.append(word)
    return vocabulary_refined


"""
Creates 2D numpy array which will hold the labels for each sample.
"""
def createMatrix(vocab, cutoff):
    return np.zeros((cutoff, len(vocab)), dtype = int)

def condenseData(m, t):
    print("condensing")
    matrix = []
    targets = []
    for i in range(len(m)):
        has1 = False
        for num in m[i]:
            if num == 1:
                has1 = True
        if has1:
            matrix.append(m[i])
            targets.append(t[i])
    return np.asarray(matrix), np.asarray(targets)



def populateTrainMatrix(vocab, matrix):
    targets = []
    for i in range(len(matrix)):
        line = file[i].split("\",\" ")
        targets.append(-1)
        try:
            for word in line[1].split():
                if word in vocab:
                    matrix[i][vocab.index(word)] = 1
            if line[2][:-2] == 'failed':
                targets[i] = 0
            else:
                targets[i] = 1
        except IndexError:
            continue
    return condenseData(matrix, targets)




def createTestMatrix(vocab, cutoff):
    return np.zeros((len(file) - cutoff, len(vocab)), dtype = int)


def populateTestMatrix(vocab, matrix, cutoff):
    targets = []
    for i in range(len(file) - len(matrix)):
        line = file[i + len(matrix)].split("\",\" ")
        targets.append(-1)
        try:
            for word in line[1].split():
                if word in vocab:
                    matrix[i][vocab.index(word)] = 1
            if line[2][:-2] == 'failed':
                targets[i] = 0
            else:
                targets[i] = 1
        except IndexError:
            continue
    return condenseData(matrix, targets)


"""
Creates keras model
"""
def createModel(cutoff, vocab_length):
    model = models.Sequential()
    
    model.add(layers.Dense(16, activation = 'relu', input_shape = (cutoff,vocab_length)))
    model.add(layers.Dense(8, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    """
    model.add(Embedding(vocab_length, 32))
    model.add(SimpleRNN(32))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    """

    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

"""
Trains keras model with the training data and evaluates with the test data
"""
def trainAndEvaluate(model, train_matrix, train_targets, test_matrix, test_targets):
    history = model.fit(train_matrix, train_targets, epochs = 15, batch_size = 500, validation_data = (train_matrix, train_targets))
    results = model.evaluate(test_matrix, test_targets)



def main():
    vocab = generateVocabulary(300)
    print(len(vocab))
    cutoff = int(len(file) * 0.7)
    train_matrix = createMatrix(vocab, cutoff)
    train_matrix, train_targets = populateTrainMatrix(vocab, train_matrix)
    print("1")

    test_matrix = createTestMatrix(vocab, cutoff)
    test_matrix, test_targets = populateTestMatrix(vocab, test_matrix, cutoff)
    print("2")

    model = createModel(cutoff, len(vocab))

    trainAndEvaluate(model, train_matrix, train_targets, test_matrix, test_targets)
    
main()