#!/usr/bin/env bash

set -e

cslc ./code_memcpy.csl --fabric-dims=17,12 --fabric-offsets=4,1 \
-o=out_code --params=width:10,height:10,zDim:10,sourceLength:10,dx:20 \
--params=srcX:0,srcY:0,MEMCPYH2D_DATA_1_ID:0,MEMCPYD2H_DATA_1_ID:1,srcZ:0,MEMCPYD2H_DATA_2_ID:2 --verbose --memcpy
cs_python ./run_memcpy.py --name out \
--iterations=10 --dx=20 --skip-compile
