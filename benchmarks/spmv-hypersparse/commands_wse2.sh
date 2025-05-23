#!/usr/bin/env bash

set -e

cslc ./src/layout.csl --arch wse2 --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=ncols:16,nrows:16,pcols:4,prows:4,max_local_nnz:8 \
--params=max_local_nnz_cols:4,max_local_nnz_rows:4,local_vec_sz:1 \
--params=local_out_vec_sz:1,y_pad_start_row_idx:4 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py --num_pe_cols=4 --num_pe_rows=4  --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --is_weight_one --run-only \
--infile_mtx=./data/rmat4.4x4.lb.mtx
