#!/usr/bin/env bash

set -e

cslc ./layout_memcpy.csl --fabric-dims=9,4 --fabric-offsets=4,1 \
--params=width:2,height:2 \
--params=LOCAL_OUT_SZ:3,LOCAL_IN_SZ:2,LAUNCH_ID:4 -o=out --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run_memcpy.py --name out -m=6 -n=4
