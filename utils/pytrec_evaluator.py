import platform
from .retrieval_score import RetrievalScore
from .helpers import dump

if not platform.system().lower().startswith("win"):
    import pytrec_eval


class MetricsEvaluator:
    def __init__(self, predicted_qrels, qrels):
        self.predicted_qrels = predicted_qrels
        self.qrels = qrels
        self.all_metrics = ["map", "ndcg_cut", "P", "recall"]
        self.final_measures = None

    def evaluate(self, metrics):
        if platform.system().lower().startswith("win"):
            print("Cannot evaluate result, windows platform.")
            return dict({"P_20": 0.1})

        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, set(self.all_metrics))
        results = evaluator.evaluate(self.predicted_qrels)

        final_measures = dict()
        for measure in metrics:
            final_measures[measure] = pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()],
            )
        self.final_measures = final_measures
        return self.final_measures

    def dump(self, filename):
        if self.final_measures is not None:
            dump(self.final_measures, filename)


def read_qrels(qrels):
    if platform.system().lower().startswith("win"):
        return dict()
    with open(qrels, "r") as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    return qrel
