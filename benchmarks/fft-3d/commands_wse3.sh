#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=N:16,NUM_PENCILS_PER_DIM:4,FP:1 --memcpy --channels=1 -o out
cs_python run.py --name out --real --norm 1
cs_python run.py --inverse --name out --norm 1
cslc --arch=wse3 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=N:16,NUM_PENCILS_PER_DIM:4,FP:0 --memcpy --channels=1 -o out
cs_python run.py --name out
cs_python run.py --inverse --name out
