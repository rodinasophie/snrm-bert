from .representations import Autoencoder
import fasttext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from datetime import datetime
import re
# TODO: use built-in embedding layers?


class Embeddings:
    def __build_emb_list(self, word_file, emb_file):
        self.model = fasttext.load_model(emb_file)
        self.dim = self.model.get_dimension() if not self.is_stub else 300
        word_ids = open(word_file, "r", encoding="utf-8")
        self.word_embeddings = []
        i = 0
        for line in word_ids:
            word_id, word = line.rstrip().split("\t")
            self.word_embeddings.append(self.model[word])
            assert i == int(word_id)
            i += 1
        print("Word_id to embedding is built", flush=True)


    def __init__(self, emb_file, word_file, is_stub):
        self.is_stub = is_stub
        if not is_stub:
            self.__build_emb_list(word_file, emb_file)
            
    def matrix(self, text, max_len, fasttext=False):
        words = text.split(" ")
        matrix = np.empty(())
        dim = self.dim

        matrix = np.zeros((max_len, dim))
        for i in range(min(len(words), max_len)):
            if words[i] == '':
                continue
            if not fasttext:
                matrix[i] = (
                    self.word_embeddings[int(words[i])] if not self.is_stub else np.random.choice(100, dim)
                )
            else:
                matrix[i] = (
                    self.model[words[i]] if not self.is_stub else np.random.choice(100, dim)
                )

        return matrix


""" Main class implementing SNRM model.

"""


class SNRM:
    def __init__(
        self,
        fembeddings,
        fwords,
        learning_rate=5e-5,
        batch_size=32,
        layers=[300, 100, 5000],
        reg_lambda=10e-7,
        drop_prob=0.6,
        qmax_len=100,
        dmax_len=1000,
        is_stub=False,
    ):
        self.reg_lambda = reg_lambda
        self.qmax_len = qmax_len
        self.dmax_len = dmax_len

        self.training_loss = 0.0
        self.validation_loss = 0.0

        self.training_steps = 0
        self.validation_steps = 0

        self.layers = layers
        self.embeddings = Embeddings(fembeddings, fwords, is_stub=is_stub)
        self.autoencoder = Autoencoder(layers, drop_prob=drop_prob)

        self.criterion = nn.MarginRankingLoss(margin=1.0)
        self.optimizer = optim.SGD(
            self.autoencoder.parameters(), lr=learning_rate, momentum=0.9
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device to use:", self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.autoencoder = nn.DataParallel(self.autoencoder)

        self.autoencoder.to(self.device)


    def __build_emb_input(self, batch, qmax_len, dmax_len):
        queries = []
        docs1 = []
        docs2 = []
        
        for triple in batch:
            q, d1, d2 = triple
            queries.append(self.embeddings.matrix(q, max_len = qmax_len, fasttext = True))
            docs1.append(self.embeddings.matrix(d1, max_len = dmax_len))
            docs2.append(self.embeddings.matrix(d2, max_len = dmax_len))
        return np.asarray(queries), np.asarray(docs1), np.asarray(docs2)


    def __reshape2_4d(self, tensor):
        return (
            torch.from_numpy(tensor)
            .float()
            .view(1, tensor.shape[2], tensor.shape[0], tensor.shape[1])
        )

    """
    An input format for training: query, doc1, doc2, y
    
    """

    def train(self, batch):
        self.autoencoder.train()
        queries, docs1, docs2 = self.__build_emb_input(
            batch, self.qmax_len, self.dmax_len
        )
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        q_out = self.autoencoder(self.__reshape2_4d(queries).to(self.device))
        d1_out = self.autoencoder(self.__reshape2_4d(docs1).to(self.device))
        d2_out = self.autoencoder(self.__reshape2_4d(docs2).to(self.device))

        reg_term = (
            torch.cat((q_out, d1_out, d2_out), dim=1)
            .sum(dim=1, keepdim=True)
            .to(self.device)
        )
        x1 = (q_out * d1_out).sum(dim=1, keepdim=True).to(self.device)
        x2 = (q_out * d2_out).sum(dim=1, keepdim=True).to(self.device)

        target = torch.ones(1).to(self.device)
        loss = self.criterion(x1, x2, target) + self.reg_lambda * reg_term

        loss.mean().backward()
        self.optimizer.step()

        self.training_loss += loss.mean().item()
        self.training_steps += 1
        return loss.mean().item()

    def validate(self, batch):
        self.autoencoder.eval()
        queries, docs1, docs2 = self.__build_emb_input(
            batch, self.qmax_len, self.dmax_len
        )

        q_out = self.autoencoder(self.__reshape2_4d(queries).to(self.device))
        d1_out = self.autoencoder(self.__reshape2_4d(docs1).to(self.device))
        d2_out = self.autoencoder(self.__reshape2_4d(docs2).to(self.device))

        reg_term = torch.cat((q_out, d1_out, d2_out), dim=1).sum(dim=1, keepdim=True)
        x1 = (q_out * d1_out).sum(dim=1, keepdim=True)
        x2 = (q_out * d2_out).sum(dim=1, keepdim=True)

        target = torch.ones(1).to(self.device)
        loss = self.criterion(x1, x2, target) + self.reg_lambda * reg_term

        self.validation_loss += loss.mean().item()
        self.validation_steps += 1
        return loss.mean().item()

    def reset_loss(self, loss):
        if loss == "train":
            self.training_loss = 0.0
            self.training_steps = 0
        elif loss == "valid":
            self.validation_loss = 0.0
            self.validation_steps = 0
        else:
            Exception("No loss found: ", loss)

    def get_loss(self, loss):
        if loss == "train":
            return self.training_loss / self.training_steps
        elif loss == "valid":
            return self.validation_loss / self.validation_steps
        else:
            Exception("No loss found: ", loss)

    def evaluate_repr(self, batch):
        repr_tensor = torch.empty(batch.shape[0], self.layers[-1])
        for i in range(batch.shape[0]):
            d_m = self.embeddings.matrix(batch[i][1], max_len=self.dmax_len)
            d_m = np.expand_dims(d_m, axis=0)
            d_out = self.autoencoder(self.__reshape2_4d(d_m).to(self.device))
            repr_tensor[i] = d_out[:, 0, :, :]
        return repr_tensor

    def save(self, filename):
        print("Saving model to ", filename)
        torch.save(self.autoencoder.state_dict(), filename)
        print("Saved.")

    def load(self, filename):
        print("Uploading model to ", filename)
        self.autoencoder.load_state_dict(torch.load(filename))
        print("Uploaded.")

    def save_checkpoint(self, filename, epoch):
        print("Saving checkpoint for epoch #", epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.autoencoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )

    """
        Uploads the old model and returns a new epoch to start training from.
    """

    def load_checkpoint(self, filename, epochs_num):
        checkpoint = torch.load(filename)
        epoch = checkpoint["epoch"]
        if epoch != epochs_num - 1:
            self.autoencoder.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            return epoch + 1, True
        return 0, False
