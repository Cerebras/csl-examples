#!/usr/bin/env bash

set -e

cslc ./code.csl --fabric-dims=6,3 \
--fabric-offsets=1,1 --params=width:4 -o out
cs_python run.py --name out
