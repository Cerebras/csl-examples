#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl --fabric-dims=11,3 \
--fabric-offsets=4,1 --params=width:4,num_elems:5 -o out  \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out