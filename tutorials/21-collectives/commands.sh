#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=15,15 --fabric-offsets=0,0 \
--params=Pw:15,Ph:15,chunk_size:3 -o out
cs_python run.py --name out
