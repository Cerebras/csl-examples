#!/usr/bin/env bash

set -e

cslc ./code.csl --fabric-dims=11,3 \
--fabric-offsets=4,1 --params=width:4 -o out  \
--params=MEMCPYH2D_DATA_1_ID:6 \
--params=MEMCPYD2H_DATA_1_ID:7 --params=LAUNCH_ID:8 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
