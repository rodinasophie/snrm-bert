import argparse
import json
from utils import EvaluationLoader
from snrm import SNRM
from utils.inverted_index import build_inverted_index
from utils.retrieval_score import RetrievalScore
from utils.pytrec_evaluator import MetricsEvaluator, read_qrels

"""
Testing and evaluating the model.
"""


def evaluate_metrics(predicted_qrels, qrels, metrics):
    evaluator = MetricsEvaluator(predicted_qrels, qrels)
    return evaluator.evaluate(metrics)


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
    # Load model from file
    eval_loader = EvaluationLoader(args.test_docs, args.test_queries)
    model.load(args.model)

    # Build inverted index
    index = build_inverted_index(
        args.batch_size, model, eval_loader, args.inverted_index
    )

    # Estimate retrieval score for each document and each query
    retrieval_score = RetrievalScore()
    predicted_qrels = retrieval_score.evaluate(
        eval_loader, index, model, args.batch_size
    )
    retrieval_score.dump(args.result_qrels)
    print(predicted_qrels)

    # Evaluate retrieval metrics
    metrics = evaluate_metrics(
        predicted_qrels, read_qrels(args.test_qrels), args.metrics
    )
    print(metrics)


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
