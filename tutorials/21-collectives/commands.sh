#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=22,17 --fabric-offsets=4,1 \
--params=Pw:15,Ph:15,chunk_size:3 -o out \
--params=LAUNCH_ID:9 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
