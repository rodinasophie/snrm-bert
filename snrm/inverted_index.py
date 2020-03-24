import json


class InvertedIndex:
    def __init__(self, out_file):
        self.index = dict()
        self.out_file = out_file

    def construct(self, doc_ids, doc_repres):
        repres_len = doc_repres.shape[1]
        for i in range(doc_repres.shape[0]):
            for j in range(repres_len):
                if doc_repres[i][j] > 0.0:
                    if j not in self.index:
                        self.index[j] = []
                    self.index[j].append([doc_ids[i], doc_repres[i][j].item()])

    def get_index(self):
        return self.index

    def flush(self):
        json_file = json.dumps(self.index)
        f = open(self.out_file, "w")
        f.write(json_file)
        f.close()

    def read_index(self):
        with open(self.out_file, "r") as f:
            self.index = json.load(f)
        return self.index

