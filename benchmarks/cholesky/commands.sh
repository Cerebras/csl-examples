#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=17,12 --fabric-offsets=4,1 \
-o out \
--params=P:10 --params=Nt:4 \
--params=LAUNCH_ID:2 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
