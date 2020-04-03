import argparse
import json
from utils import EvaluationLoader
from snrm import SNRM
from snrm.inverted_index import build_inverted_index
from utils.evaluation_metrics import retrieval_score

"""
Testing and evaluating the model.
"""


def evaluate_metrics(model, eval_loader, metrics, index, batch_size):
    queries_len = eval_loader.queries_length()
    offset = 0
    res = dict()
    while offset < queries_len:
        queries = eval_loader.generate_queries(size=batch_size)
        qreprs = model.evaluate_repr(queries)
        for qrepr, q in zip(qreprs, queries):
            res[q] = retrieval_score(qrepr, index)
        offset += batch_size
    return res


def run(args):
    model = SNRM(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        layers=args.layers,
        reg_lambda=args.reg_lambda,
        drop_prob=args.drop_prob,
        fembeddings=args.embeddings,
        qmax_len=args.qmax_len,
        dmax_len=args.dmax_len,
        is_stub=args.is_stub,
    )
    eval_loader = EvaluationLoader(args.test_docs, args.test_queries, args.test_qrels)

    model.load(args.model)
    index = build_inverted_index(
        args.batch_size, model, eval_loader, args.inverted_index
    )
    results = evaluate_metrics(model, eval_loader, args.metrics, index, args.batch_size)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--params", type=str, help="Path to json-file with params"
    )
    args, _ = parser.parse_known_args()
    with open(args.params) as f:
        params = json.load(f)
    for key, val in params.items():
        parser.add_argument("--" + key, default=val)
    args = parser.parse_args()
    print(args)
    run(args)
