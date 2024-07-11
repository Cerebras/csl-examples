#!/usr/bin/env bash

set -e

cslc ./layout.csl --arch wse2 --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=ncols:16 --params=nrows:16 --params=pcols:4 --params=prows:4 --params=max_local_nnz:8 \
--params=max_local_nnz_cols:4 --params=max_local_nnz_rows:4 --params=local_vec_sz:1 \
--params=local_out_vec_sz:1 --params=y_pad_start_row_idx:4 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py --num_pe_cols=4 --num_pe_rows=4  --latestlink out --channels=1 --width-west-buf=0 \
--width-east-buf=0 --is_weight_one --run-only --infile_mtx=./data/rmat4.4x4.lb.mtx
