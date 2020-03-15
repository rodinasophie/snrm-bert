class InvertedIndexConstructor:
    def __init__(self, out_file):
        self.index = dict()
        self.out_file = out_file

    def construct(self, doc_ids, batch):
        repres_len = batch.shape[1]
        for i in range(batch.shape[0]):
            for j in range(repres_len):
                if batch[i][j] > 0.0:
                    if j not in self.index:
                        self.index[j] = []
                    self.index[j].append((doc_ids[i], batch[i][j]))

    def get_index(self):
        return self.index
