#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--colors=x_in:1,ax_out:3,b_in:4 -o out \
--params=kernel_rows:4,kernel_cols:4,matrix_rows:32,matrix_cols:16 \
--params=MEMCPYH2D_DATA_1_ID:10 --params=MEMCPYH2D_DATA_2_ID:11 \
--params=MEMCPYD2H_DATA_1_ID:12 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out
