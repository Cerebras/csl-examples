#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./code.csl --fabric-dims=11,6 --fabric-offsets=4,1 -o out \
--params=MEMCPYD2H_DATA_1_ID:1 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
