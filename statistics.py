import argparse
import json
from snrm import SNRM
from utils.helpers import manage_model_params
from utils.helpers import path_exists, load_file, list_files, join, estimate_sparsity
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

VERY_LARGE_EPOCH = 1000
MAX_ITER = 20
def count_zeros(model, loader, batch_size, element_type):
    is_end = False
    print("Counting zeros for ", element_type)
    fn = loader.generate_queries if element_type == 'queries' else loader.generate_docs
    zeros = 0
    counter = 0
    all_zeros=0
    k = 0
    while not is_end:
        if k == MAX_ITER:
            break
        print(k)
        k += 1
        ids, docs, is_end = fn(batch_size)
        repr = model.evaluate_repr(docs, input_type=element_type).cpu().numpy()
        counter += repr.shape[0]
        for i in range(repr.shape[0]):
            z, nans = estimate_sparsity(repr[i])
            if nans != 0:
                print("Nans for id {} = {}, zeros = {}".format(ids[i], nans, z), flush=True)
            zeros += z
            if z == repr.shape[1]:
                all_zeros += 1
                #print("All zeros({}, {}) for q: id-{}, text-{}".format(z, repr.shape[1], ids[i], docs[i]), flush=True) 
        if is_end:
            break
    print("All zeros: ", all_zeros)
    loader.reset()
    return float(zeros)/float(counter) # return mean zeros
        

def check_sparsity(model, model_epoch_pth, loader, batch_size):
    epoch, is_resumed = model.load_checkpoint(model_epoch_pth, VERY_LARGE_EPOCH)
    assert is_resumed == True
    epoch -= 1 # real learned epoch
    print("Checking sparsity for epoch #", epoch, flush=True)
    query_mean_zeros = count_zeros(model, loader, batch_size, 'queries')
    docs_mean_zeros = count_zeros(model, loader, batch_size, 'docs')

    return query_mean_zeros, docs_mean_zeros, epoch 


def run(args, model_params):
    print("\nRunning statistics for {} ...".format(model_params["model_name"]))

    model = SNRM(
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        layers=model_params["layers"],
        reg_lambda=model_params["reg_lambda"],
        drop_prob=model_params["drop_prob"],
        fembeddings=args.embeddings,
        fwords=args.words,
        qmax_len=args.qmax_len,
        dmax_len=args.dmax_len,
        is_stub=args.is_stub,
    )
    
    docs_dict = dataset.data_loader.load_docs(args.docs)

    valid_loader = dataset.data_loader.DataLoader(
        args.valid_queries, args.valid_qrels, docs_dict,
    )

    writer = SummaryWriter(args.summary_folder)
    files = list_files(model_params["dir"])
    files.sort()
    print(files)
    #files = ["model_checkpoint_epoch1.pth", "model_checkpoint_epoch19.pth"]
    for f in files:
        if f.startswith("model_checkpoint_epoch"):
            start = datetime.now()
            qmean, dmean, epoch = check_sparsity(model, join(model_params["dir"], f), valid_loader, model_params["batch_size"])
            time = datetime.now() - start
            print("Mean sparsity for epoch {}: queries sparsity = {}, docs sparsity = {}, execution time: {}".format(epoch, qmean, dmean, time), flush=True)
            writer.add_scalars( model_params["model_name"]+"-sparsity", {"Query sparsity rate": qmean, "Docs sparsity rate": dmean}, epoch)

    writer.close()
    print("Finished gathering statistics for {}".format(model_params["model_name"]))




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
