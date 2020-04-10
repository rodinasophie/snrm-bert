from utils.pytrec_evaluator import MetricsEvaluator
from utils.retrieval_score import RetrievalScore
from utils.inverted_index import build_inverted_index, load_inverted_index
from utils.pytrec_evaluator import read_qrels


"""
Testing and evaluating the model.
"""


def _evaluate_metrics(predicted_qrels, qrels, metrics):
    evaluator = MetricsEvaluator(predicted_qrels, qrels)
    return evaluator.evaluate(metrics)


"""
    Evaluate retrieval score.
    Returns predicted ranking for each query.
"""


def _compute_retrieval_score(model_params, model, eval_loader, index, dump):
    retrieval_score = RetrievalScore()
    predicted_qrels = retrieval_score.evaluate(
        eval_loader, index, model, model_params["batch_size"]
    )
    if dump:
        print("Dumping retrieval_score to ", model_params["retrieval_score"])
        retrieval_score.dump(model_params["retrieval_score"])
    return predicted_qrels


"""
    Build inverted index and evaluate model according to the given metrics.
"""


def evaluate_model(
    model_params, model, eval_loader, test_metrics, dump, rebuild_inverted_index=True,
):
    # Build inverted index
    if rebuild_inverted_index:
        index = build_inverted_index(
            model_params["batch_size"],
            model,
            eval_loader,
            model_params["inverted_index"],
            dump=dump,
        )
    else:
        index = load_inverted_index(model_params["inverted_index"])

    # Estimate retrieval score for each document and each query
    predicted_qrels = _compute_retrieval_score(
        model_params, model, eval_loader, index, dump=dump
    )
    # print(predicted_qrels)

    # Evaluate retrieval metrics
    metrics = _evaluate_metrics(
        predicted_qrels, eval_loader.generate_qrels(), test_metrics
    )

    return metrics
