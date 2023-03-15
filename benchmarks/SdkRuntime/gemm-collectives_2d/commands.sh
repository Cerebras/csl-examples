#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=11,6 \
--fabric-offsets=4,1 \
--memcpy --channels=1 -o out
cs_python run.py --name out
