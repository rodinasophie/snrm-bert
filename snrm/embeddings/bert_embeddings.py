import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def generate_dict(fword):
    word_ids = open(fword, "r", encoding="utf-8")
    word_dict = dict()
    for line in word_ids:
        word_id, word = line.rstrip().split("\t")
        word_dict[int(word_id)] = word
    word_ids.close()
    return word_dict


class BertEmbeddings:
    def __init__(self, fword, emb_type):
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        emb_type_to_len = {"cat": 3072, "sum": 768}

        self.emb_type = emb_type
        self.dim = emb_type_to_len[self.emb_type]

        self.word_dict = generate_dict(fword)
        print("BERT_EMB: Word dict is generated.", flush=True)

    def __split_and_mark(self, text):
        ids = text.rstrip().split(" ")
        words = [self.word_dict[int(id_)] for id_ in ids if id_ != ""]
        return "[CLS] " + " ".join(words) + " [SEP]"

    def __concat_emb(self, token_embeddings):
        token_vecs_cat = []
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs_cat.append(cat_vec)
        return token_vecs_cat

    def __sum_emb(self, token_embeddings):
        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)
        return token_vecs_sum

    def __build_token_embeddings(self, marked_text):
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        return token_embeddings

    def __build_emb(self, text):
        marked_text = self.__split_and_mark(text)
        token_embeddings = self.__build_token_embeddings(marked_text)
        if self.emb_type == "cat":
            return self.__concat_emb(token_embeddings)
        else:
            return self.__sum_emb(token_embeddings)

    def matrix(self, texts, max_len):
        matrix = np.zeros((len(texts), max_len, self.dim))
        for t in range(len(texts)):
            m = self.__build_emb(texts[t])
            for i in range(min(len(m), max_len)):
                matrix[t][i] = m[i]
        return matrix

    def get_emb_len(self):
        return self.dim
