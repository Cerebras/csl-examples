#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=11,3 \
--fabric-offsets=4,1 --params=MEMCPYD2H_DATA_1_ID:1 \
--memcpy --channels=1 -o out
cs_python run.py --name out
