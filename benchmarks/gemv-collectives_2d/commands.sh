#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=kernel_rows:4,kernel_cols:4,matrix_rows:32,matrix_cols:16 \
--memcpy --channels=1 -o out
cs_python run.py --name out
