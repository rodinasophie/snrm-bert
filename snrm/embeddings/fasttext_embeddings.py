import fasttext
import numpy as np


class FastTextEmbeddings:
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

    def matrix(self, texts, max_len):
        matrix = np.zeros((len(texts), max_len, self.dim))
        for t in range(len(texts)):
            words = texts[t].split(" ")
            for i in range(min(len(words), max_len)):
                if words[i] == "":
                    continue
                matrix[t][i] = self.word_embeddings[int(words[i])]

        return matrix

    def get_emb_len(self):
        return self.dim
