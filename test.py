import argparse
import json
from snrm import SNRM
from utils.helpers import manage_model_params
from utils.evaluation_helpers import evaluate_model
from utils.helpers import path_exists, load_file


def check_rebuild(model_params):
    rebuild_inverted_index = True
    rebuild_retrieval_score = True

    if path_exists(model_params["inverted_index"]) and not args.rerun_if_exists:
        print("Inverted index already exists, won't be rebuild.")
        rebuild_inverted_index = False

    if path_exists(model_params["retrieval_score"]) and not args.rerun_if_exists:
        print("Retrieval score already exists, won't be rebuild.")
        rebuild_retrieval_score = False

    return rebuild_inverted_index, rebuild_retrieval_score


def run(args, model_params):
    print("\nRunning training for {} ...".format(model_params["model_name"]))

    rebuild_inverted_index, rebuild_retrieval_score = check_rebuild(model_params)

    if not rebuild_inverted_index and not rebuild_retrieval_score:
        metrics = load_file(model_params["final_metrics"])
        print(metrics)
        return

    model = SNRM(
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        layers=model_params["layers"],
        reg_lambda=model_params["reg_lambda"],
        drop_prob=model_params["drop_prob"],
        fembeddings=args.embeddings,
        fwords = args.words,
        qmax_len=args.qmax_len,
        dmax_len=args.dmax_len,
        is_stub=args.is_stub,
    )

    # Load model from file
    eval_loader = dataset.evaluation_loader.EvaluationLoader(args.test_docs, args.test_queries, args.test_qrels)
    model.load(model_params["model_pth"])

    # Evaluate model
    # If we need to rebuild inverted index, we also need to rebuild the retrieval score.
    # If inverted index is built and we are on this stage, we need to rebuild the retrieval score anyway.
    metrics = evaluate_model(
        model_params,
        model,
        eval_loader,
        args.test_metrics,
        dump=True,
        rebuild_inverted_index=rebuild_inverted_index,
    )

    print(metrics)
    print("Finished testing for {}".format(model_params["model_name"]))


def setup(module):
    global dataset
    dataset = __import__(module, fromlist=["object"])

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

    setup(".".join(["utils", args.dataset]))
    models_to_train = list(args.models)
    for model in models_to_train:
        manage_model_params(args, model)
        run(args, model)
