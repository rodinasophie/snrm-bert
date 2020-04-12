#!/bin/bash

source /home/mrim/rodinas/snrmenv/bin/activate
python train.py --params=params/params_stub.json
python test.py --params=params/params_stub.json
deactivate
