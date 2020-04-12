#!/bin/bash

source /home/mrim/rodinas/snrmenv/bin/activate
python train.py --params=params/params_msmarco_docs.json
python test.py --params=params/params_msmarco_docs.json
deactivate
