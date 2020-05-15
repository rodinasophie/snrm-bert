#!/bin/bash

source /home/mrim/rodinas/snrmenv/bin/activate
python train.py --params=params/passages/params_10000.json
deactivate
