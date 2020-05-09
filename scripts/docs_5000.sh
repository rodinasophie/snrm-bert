#!/bin/bash

source /home/mrim/rodinas/snrmenv/bin/activate
python train.py --params=params/docs/params_5000.json
deactivate
