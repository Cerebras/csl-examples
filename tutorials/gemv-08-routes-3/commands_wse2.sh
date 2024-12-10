#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl --fabric-dims=11,5 \
--fabric-offsets=4,1 --params=kernel_x_dim:4,kernel_y_dim:3,M:6,N:8 \
-o out --memcpy --channels 1
cs_python run.py --name out
