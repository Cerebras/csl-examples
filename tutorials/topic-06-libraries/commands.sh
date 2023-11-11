#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 \
--params=iterations:200 -o out \
--params=MEMCPYD2H_DATA_1_ID:1 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out --tolerance 0.1
