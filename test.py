import argparse
import json
from utils import EvaluationLoader
from snrm import SNRM
from utils.inverted_index import build_inverted_index
from utils.retrieval_score import RetrievalScore
from utils.pytrec_evaluator import MetricsEvaluator, read_qrels
from utils.manage_model import manage_model_params

"""
Testing and evaluating the model.
"""


def evaluate_metrics(predicted_qrels, qrels, metrics):
    evaluator = MetricsEvaluator(predicted_qrels, qrels)
    return evaluator.evaluate(metrics)


def run(args, model_params):
    model = SNRM(
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        layers=model_params["layers"],
        reg_lambda=model_params["reg_lambda"],
        drop_prob=model_params["drop_prob"],
        fembeddings=args.embeddings,
        qmax_len=args.qmax_len,
        dmax_len=args.dmax_len,
        is_stub=args.is_stub,
    )
    # Load model from file
    eval_loader = EvaluationLoader(args.test_docs, args.test_queries)
    model.load(model_params["model_pth"])

    # Build inverted index
    index = build_inverted_index(
        model_params["batch_size"], model, eval_loader, model_params["inverted_index"]
    )

    # Estimate retrieval score for each document and each query
    retrieval_score = RetrievalScore()
    predicted_qrels = retrieval_score.evaluate(
        eval_loader, index, model, model_params["batch_size"]
    )
    retrieval_score.dump(model_params["retrieval_score"])
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

    models_to_train = list(args.models)
    for model in models_to_train:
        manage_model_params(args, model)
        run(args, model)
