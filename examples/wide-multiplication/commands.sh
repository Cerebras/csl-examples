#!/usr/bin/env bash

set -e

cslc ./code.csl --fabric-dims=5,5 --fabric-offsets=2,2 \
--params=numBits:256 --colors=recvColor:0,outputColor:1,triggerColor:43 -o out
cs_python run.py --name out
