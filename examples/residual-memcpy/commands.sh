#!/usr/bin/env bash
cslc ./layout_memcpy.csl --fabric-dims=9,4 --fabric-offsets=4,1 \
--params=width:2,MEMCPYH2D_DATA_1_ID:0,MEMCPYH2D_DATA_2_ID:1,height:2,MEMCPYH2D_DATA_3_ID:2,MEMCPYD2H_DATA_1_ID:3 \
--params=LOCAL_OUT_SZ:3,LOCAL_IN_SZ:2 -o=out --memcpy
cs_python run_memcpy.py --name out -m=6 -n=4
