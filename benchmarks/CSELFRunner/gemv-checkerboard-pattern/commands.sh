#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=6,6 --fabric-offsets=1,1 \
--colors=x_in:1,y_out:2,ax_out:3,b_in:4,sentinel:43 -o out
cs_python run.py --name out
