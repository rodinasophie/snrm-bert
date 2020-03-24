from .representations import Autoencoder
import fasttext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# TODO: use built-in embedding layers?


class Embeddings:
    def __init__(self, emb_file):
        self.model = fasttext.load_model(emb_file)

    def matrix(self, text, max_len):
        words = text.split()
        matrix = np.empty(())
        dim = self.model.get_dimension()
        matrix = np.zeros((max_len, dim))
        for i in range(min(len(words), max_len)):
            matrix[i] = self.model[words[i]]
        return matrix


""" Main class implementing SNRM model.

"""


class SNRM:
    def __init__(
        self,
        fembeddings,
        learning_rate=5 * 10e-5,
        batch_size=32,
        layers=[300, 100, 5000],
        reg_lambda=10e-7,
        drop_prob=0.6,
        qmax_len=100,
        dmax_len=1000,
    ):
        self.reg_lambda = reg_lambda
        self.qmax_len = qmax_len
        self.dmax_len = dmax_len

        self.layers = layers
        self.embeddings = Embeddings(fembeddings)
        self.autoencoder = Autoencoder(layers, drop_prob=drop_prob)

        self.criterion = nn.MarginRankingLoss(margin=1.0)
        self.optimizer = optim.SGD(
            self.autoencoder.parameters(), lr=learning_rate, momentum=0.9
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.autoencoder = nn.DataParallel(self.autoencoder)

        self.autoencoder.to(self.device)

    def __build_emb_input(self, batch, qmax_len, dmax_len):
        output = []
        for triple in batch:
            q, d1, d2 = triple
            q_m = self.embeddings.matrix(q, max_len=qmax_len)
            d1_m = self.embeddings.matrix(d1, max_len=dmax_len)
            d2_m = self.embeddings.matrix(d2, max_len=dmax_len)
            output.append(np.array([q_m, d1_m, d2_m]))
        return np.asarray(output)

    def __reshape2_4d(self, tensor):
        return (
            torch.from_numpy(tensor)
            .float()
            .view(1, tensor.shape[1], 1, tensor.shape[0])
        )

    """
    An input format for training: query, doc1, doc2, y
    
    """

    def train(self, batch):
        running_loss = 0.0
        out_batch = self.__build_emb_input(batch, self.qmax_len, self.dmax_len)
        batch_size = out_batch.shape[0]
        for i in range(batch_size):
            # get the inputs; data is a list of [inputs, labels]
            query, d1, d2 = out_batch[i]

            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize

            q_out = self.autoencoder(self.__reshape2_4d(query).to(self.device))
            d1_out = self.autoencoder(self.__reshape2_4d(d1).to(self.device))
            d2_out = self.autoencoder(self.__reshape2_4d(d2).to(self.device))

            reg_term = torch.cat((q_out, d1_out, d2_out), dim=1).sum(
                dim=1, keepdim=True
            )
            x1 = (q_out * d1_out).sum(dim=1, keepdim=True)
            x2 = (q_out * d2_out).sum(dim=1, keepdim=True)

            target = torch.ones(1).to
            loss = self.criterion(x1, x2, target) + self.reg_lambda * reg_term
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

        print("Finished Training")
        return running_loss / batch_size, 0.0  # FIXME: validation loss instead of 0.0

    def evalute_repr(self, batch):
        repr_tensor = torch.empty(batch.shape[0], self.layers[-1])
        for i in range(batch.shape[0]):
            d_m = self.embeddings.matrix(batch[i][1], max_len=self.dmax_len)
            d_out = self.autoencoder(self.__reshape2_4d(d_m).to(self.device))
            repr_tensor[i] = (batch[i][0], d_out)
        return repr_tensor

    def save(self, filename):
        print("Saving model...")
        self.autoencoder.save(filename)
        print("Saved.")
