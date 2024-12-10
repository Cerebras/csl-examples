#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=P:4,Mt:14,Kt:14,Nt:14 \
--memcpy --channels=1 -o out
cs_python run.py --name out
