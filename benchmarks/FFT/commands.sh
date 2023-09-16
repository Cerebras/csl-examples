#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 \
--params=DIM:1,Nz:4,FP:2 --colors=LAUNCH:8 --memcpy --channels=1 -o out-1D
cs_python run.py --name out-1D --fabric-offsets=4,1
cs_python run.py --inverse --name out-1D --fabric-offsets=4,1
cslc ./layout.csl --fabric-dims=11,3 --fabric-offsets=4,1 \
--params=DIM:2,Nz:4,FP:1 --colors=LAUNCH:8 --memcpy --channels=1 -o out-2D
cs_python run.py --name out-2D --fabric-offsets=4,1
cs_python run.py --inverse --name out-2D --fabric-offsets=4,1
