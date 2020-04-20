#!/bin/bash

source /home/mrim/rodinas/snrmenv/bin/activate
python test.py --params=params/params_msmarco_docs.json
deactivate
