#!/usr/bin/env bash

set -e

cslc ./code.csl --fabric-dims=8,12 \
--fabric-offsets=4,1 -o out \
--params=size:10 \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYD2H_DATA_1_ID:1 --params=LAUNCH_ID:2 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
