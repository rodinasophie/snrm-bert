
import platform
from .retrieval_score import RetrievalScore
if not platform.system().lower().startswith('win'):
    import pytrec_eval


class MetricsEvaluator:
    def __init__(self, predicted_qrels, qrels):
        self.predicted_qrels = predicted_qrels
        self.qrels = qrels

    def evaluate(self, metrics):
        if platform.system().lower().startswith('win'):
            return "Cannot evaluate result, windows platform."
        
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, metrics)
        results = evaluator.evaluate(self.predicted_qrels)
        # FIXME: don't return all values for all queries,
        # return only one value for each metric
        return results


def read_qrels(qrels):
    if platform.system().lower().startswith('win'):
        return dict()
    with open(qrels, "r") as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    return qrel
