#!/usr/bin/env bash

set -e

cslc ./layout.csl --arch=wse2 --fabric-dims=9,4 --fabric-offsets=4,1 \
--params=width:2,height:2 \
--params=LOCAL_OUT_SZ:3,LOCAL_IN_SZ:2 -o=out --memcpy --channels=1 \
--width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
