# Standalone Neural Ranking Model with Bert modification

## Motivation
The main idea of that repo is to extend the SNR model proposed by H.Zamani et al. [1] with the Bert[2] embeddings and apply the model to the MSMARCO dataset[3].


## Code structure

* `snrm\` - the main folder with PyTorch code
* `snrm\embeddings\` - fasttext embeddings and Bert embeddings
* `snrm\representations\` - autoencoder architecture
* `params\` - the folder to keep param-files
* `scripts\` - the folder to keep run scripts(.sh)
* `utils\` - the rest part of code - data loaders, inverted index, retrieval score

## Vocabulary and data representation

The input data consists of documents file, query and qrel files(for training, testing and validation).
To save the memory, the one vocabulary over all documents and queries is built. The document file and query file is considered to contain word ids instead of words itself.
To get more information, please, see `stub_data/`, `utils/stub` or `utils/msmarco`. 


## Stages
There are two main stages considered to create the model:
* Training stage - training the model on training data
* Test stage - testing the model and estimating the result
The final minor stage:
* Gather the statistics - estimate the sparsity rate of the text and query representation


### Parameters
All `train.py`, `test.py` and `statistics.py` files need a unified `.json` parameter's input file.
Here is an example of the file:
```{
    "dataset": "msmarco",
    "docs": "PATH/TO/DOCUMENTS",
    "words": "PATH/TO/VOCABULARY",
    "train_queries": "PATH/TO/TRAIN/QUERIES",
    "train_qrels": "PATH/TO/TRAIN/QRELS",
    "test_queries": "PATH/TO/TEST/QUERIES",
    "test_qrels": "PATH/TO/TEST/QRELS",
    "valid_queries": "PATH/TO/VALIDATION/QUERIES",
    "valid_qrels": "PATH/TO/VALIDATION/QRELS",
    "embeddings": "PATH/TO/EMBEDDINGS",
    "summary_folder": "./passages-summary/",
    "qmax_len": 10,
    "dmax_len": 300,
    "epochs": 3,
    "valid_metric": "P_20",
    "test_metrics": [
        "P_5",
        "P_10",
        "P_20",
        "ndcg_cut_5",
        "ndcg_cut_10",
        "ndcg_cut_20",
        "ndcg_cut_1000",
        "map",
        "recall_1000"
    ],
    "is_stub": 0, # always 0
    "rerun_if_exists": 0,
    "inverted_index": "PATH/TO/INVERTED/INDEX/FILE/TO/GENERATE",
    "retrieval_score": "PATH/TO/WRITE/THE/RETRIEVAL/SCORE",
    "final_metrics": "PATH/TO/REPORT/FINAL/METRICS",
    "model_pth": "model.pth",
    "model_checkpoint_pth": "model_checkpoint.pth",
    "models_folder": "./passages-models/",
    "models": [
        {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "layers": [
                300,
                300,
                500,
                10000
            ],
            "reg_lambda": 1e-9,
            "drop_prob": 0
        }] ```


## Usage

To run the training stage:
```
python train.py --params=param_file
```

To run the test stage:
```
python test.py --params=param_file
```

To run the statistics(estimate sparsity):
```
python statistics.py --params=param_file
```

## Customization

To customize the model to your own data, you need to add your own folder to `utils/` e.g. `utils/my_dataset` and implement your versions of `data_loader.py`, `evaluation_loader.py`, `train_loader.py`. Then state the name of your folder in the params file in the field `dataset`.

## References
[1] Hamed  Zamani,  Mostafa  Dehghani,  W.  Croft,  Erik  Learned-Miller,  and  Jaap  Kamps.“From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for In-verted Indexing”. In: Oct. 2018, pp. 497–506.doi:10.1145/3269206.3271800.
[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. “Bert: Pre-trainingof deep bidirectional transformers for language understanding”. In:arXiv preprint arXiv:1810.04805(2018).
[3] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder,and Li Deng. “MS MARCO: a human-generated machine reading comprehension dataset”.In: (2016).