from .representations import Autoencoder
import fasttext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from datetime import datetime
import random
import re

# TODO: use built-in embedding layers?


class Embeddings:
    def __build_emb_list(self, word_file, emb_file):
        if not self.is_stub:
            self.model = fasttext.load_model(emb_file)
        self.dim = self.model.get_dimension() if not self.is_stub else 300
        word_ids = open(word_file, "r", encoding="utf-8")
        self.word_embeddings = []
        i = 0
        for line in word_ids:
            word_id, word = line.rstrip().split("\t")
            self.word_embeddings.append(
                self.model[word]
                if not self.is_stub
                else np.random.normal(0, 1.0, self.dim)
            )
            assert i == int(word_id)
            i += 1
        print("Word_id to embedding is built", flush=True)

    def __init__(self, emb_file, word_file, is_stub):
        self.is_stub = is_stub
        self.__build_emb_list(word_file, emb_file)

    def matrix2(self, texts, max_len):
        matrix = np.zeros((len(texts), max_len, self.dim))
        for t in range(len(texts)):
            words = texts[t].split(" ")
            for i in range(min(len(words), max_len)):
                if words[i] == "":
                    continue
                matrix[t][i] = self.word_embeddings[int(words[i])]

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
        random.seed(43)
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
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(), lr=learning_rate
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device to use:", self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.autoencoder = nn.DataParallel(self.autoencoder)

        self.autoencoder.to(self.device)

    def __build_emb_input2(self, batch, qmax_len, dmax_len):
        queries = batch[0]
        docs1 = batch[1]
        docs2 = batch[2]

        q_emb = self.embeddings.matrix2(queries, max_len=qmax_len)
        doc1_emb = self.embeddings.matrix2(docs1, max_len=dmax_len)
        doc2_emb = self.embeddings.matrix2(docs2, max_len=dmax_len)

        return q_emb, doc1_emb, doc2_emb


    def __reshape2_4d(self, tensor):
        tensor = np.asarray([tensor[i].transpose() for i in range(tensor.shape[0])])
        return torch.from_numpy(tensor).float()
        

    """
    An input format for training: query, doc1, doc2, y
    
    """

    def train(self, batch):
        self.autoencoder.train()
        queries, docs1, docs2 = self.__build_emb_input2(
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
        target = torch.ones(q_out.shape[0]).to(self.device)
        loss = self.criterion(x1, x2, target) + self.reg_lambda * reg_term
        loss.mean().backward()
        self.optimizer.step()

        self.training_loss += loss.mean().item()
        self.training_steps += 1
        return loss.mean().item()

    def validate(self, batch):
        self.autoencoder.eval()
        queries, docs1, docs2 = self.__build_emb_input2(
            batch, self.qmax_len, self.dmax_len
        )

        q_out = self.autoencoder(self.__reshape2_4d(queries).to(self.device))
        d1_out = self.autoencoder(self.__reshape2_4d(docs1).to(self.device))
        d2_out = self.autoencoder(self.__reshape2_4d(docs2).to(self.device))

        reg_term = torch.cat((q_out, d1_out, d2_out), dim=1).sum(dim=1, keepdim=True)
        x1 = (q_out * d1_out).sum(dim=1, keepdim=True)
        x2 = (q_out * d2_out).sum(dim=1, keepdim=True)

        target = torch.ones(q_out.shape[0]).to(self.device)
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

    
    def evaluate_repr(self, batch, input_type):
        max_len = self.qmax_len if input_type == "queries" else self.dmax_len
        emb = self.embeddings.matrix2(batch, max_len=max_len)
        
        self.autoencoder.eval()
        d_out = self.autoencoder(self.__reshape2_4d(emb).to(self.device))
        
        return d_out[:, :, 0].detach()

    def save(self, filename):
        print("Saving model to ", filename)
        torch.save(self.autoencoder.state_dict(), filename)
        print("Saved.")

    def load(self, filename):
        print("Uploading model to ", filename)
        self.autoencoder.load_state_dict(torch.load(filename))
        print("Uploaded.")

    def save_checkpoint(self, filename, epoch):
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
        if epoch < epochs_num - 1:
            self.autoencoder.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            return epoch + 1, True
        return 0, False
