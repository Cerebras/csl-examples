#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=10,5 --fabric-offsets=4,1 \
--params=MEMCPYH2D_DATA_1_ID:1,MEMCPYH2D_DATA_2_ID:2,MEMCPYH2D_DATA_3_ID:4 \
--params=MEMCPYD2H_DATA_1_ID:3,MEMCPYD2H_DATA_2_ID:5,size:3 \
-o=out --memcpy --verbose
cs_python ./run.py --name out --size 3 --iters 4
