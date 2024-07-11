#!/usr/bin/env bash

set -e

cslc ./layout.csl --arch=wse2 --fabric-dims=17,12 --fabric-offsets=4,1 \
-o=out_code --params=width:10,height:10,zDim:10,sourceLength:10,dx:20 \
--params=srcX:0,srcY:0,srcZ:0 --verbose --memcpy --channels=1 \
--width-west-buf=0 --width-east-buf=0
cs_python run.py --name out \
--iterations=10 --dx=20 --skip-compile
