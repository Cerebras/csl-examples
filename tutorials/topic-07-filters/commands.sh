#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=11,5 --fabric-offsets=4,1 -o out \
--params=MEMCPYH2D_DATA_1_ID:3 \
--params=MEMCPYD2H_DATA_1_ID:4 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
