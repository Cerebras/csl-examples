#!/usr/bin/env bash
cslc ./code.csl --fabric-dims=5,5 --fabric-offsets=2,2 \
--params=DIM:1,Nz:4,FP:2 --colors=output_color:8 -o out-1D
cs_python run.py --name out-1D --fabric-offsets=2,2
cslc ./code.csl --fabric-dims=8,5 --fabric-offsets=2,2 \
--params=DIM:2,Nz:4,FP:1 --colors=output_color:8 -o out-2D
cs_python run.py --name out-2D --fabric-offsets=2,2
