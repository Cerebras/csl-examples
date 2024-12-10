#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=19,14 --fabric-offsets=4,1 \
--params=x_dim:12,y_dim:12 --memcpy --channels=1 -o out
cs_python run.py --name out --initial-state glider --iters 20
