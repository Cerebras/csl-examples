#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=4,4 \
--fabric-offsets=1,1 \
--params=LOCAL_OUT_SZ:3,LOCAL_IN_SZ:2 -o out
cs_python run.py --name out -m=6 -n=4
