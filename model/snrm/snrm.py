from model.representations import AutoEncRepresentation, BertRepresentation

""" Main class implementing SNRM model.

"""


class SNRM:
    REPRESENTATIONS = {"auto": AutoEncRepresentation(), "bert": BertRepresentation()}

    def __init__(self, reprs="auto"):
        self.reprs = [self.reprs, SNRM.REPRESENTATIONS[reprs]]
        print("SNRM constructor")

    """
    An input format for training: query, doc1, doc2, y
    
    """

    def train(self, X):
        if self.reprs[0] == "auto":
            self.reprs[1].train(X)  # train autoencoder on given data
        print("Train model")
        pass

    def save(self, filename):
        print("Saving model...")
        pass
