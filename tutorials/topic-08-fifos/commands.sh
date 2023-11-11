#!/usr/bin/env bash

set -e

cslc ./layout.csl \
--fabric-dims=10,3 --fabric-offsets=4,1 \
--params=num_elements_to_process:2048 \
-o out \
--params=MEMCPYH2D_DATA_1_ID:4 \
--params=MEMCPYD2H_DATA_1_ID:5 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
