#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl \
--fabric-dims=8,3 --fabric-offsets=4,1 \
--params=num_elems_to_process:512 \
--params=MEMCPYH2D_DATA_1_ID:4 \
--params=MEMCPYD2H_DATA_1_ID:5 \
-o out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
