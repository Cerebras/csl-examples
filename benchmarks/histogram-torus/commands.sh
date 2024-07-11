#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./code.csl \
--params=INPUT_SIZE:16,HIST_WIDTH:8,HIST_HEIGHT:8,NUM_BUCKETS:4,BUCKET_SIZE:2 \
--colors=OUT_COLOR:8 \
--fabric-dims=15,10 --fabric-offsets=4,1 -o out \
--params=MEMCPYH2D_DATA_1_ID:10 \
--params=MEMCPYD2H_DATA_1_ID:11 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
