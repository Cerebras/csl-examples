#!/usr/bin/env bash

set -e

cslc ./code.csl \
--params=INPUT_SIZE:16,HIST_WIDTH:8,HIST_HEIGHT:8,NUM_BUCKETS:4,BUCKET_SIZE:2 \
--colors=OUT_COLOR:8 \
--fabric-dims=10,10 --fabric-offsets=1,1 -o out
cs_python run.py --name out
