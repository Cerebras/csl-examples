#!/usr/bin/env bash

set -e

cslc ./bw_sync_layout.csl --arch wse2 --fabric-dims=12,7 --fabric-offsets=4,1 \
--params=width:5,height:5,pe_length:5 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=5 -n=5 -k=5 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --run-only --loop_count=1
