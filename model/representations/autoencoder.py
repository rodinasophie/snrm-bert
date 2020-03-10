from torchnlp.word_to_vector import FastText
from torch import nn


"""
Autoencoder neural network to learn document representation.

"""


class Autoencoder(nn.Model):
    def __init__(self):
        pass

    def forward(x):
        return -1


class AutoencoderRepresentation:
    def __init__(self):
        self.fastText = FastText()

    def __get_embedding(self, word):
        return self.fastText[word]

    def train(self, X):
        embeddingMatrix = []
        for x in range(len(X)):
            embeddingMatrix.append(self.__get_embedding(x))

