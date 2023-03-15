#!/usr/bin/env bash

set -e

cslc ./code.csl --fabric-dims=3,3 \
--fabric-offsets=1,1 --params=useIntegerType:0 -o out
cs_python run.py --name out
