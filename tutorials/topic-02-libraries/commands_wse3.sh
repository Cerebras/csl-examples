#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 \
--params=iterations:200 -o out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out --tolerance 0.1
