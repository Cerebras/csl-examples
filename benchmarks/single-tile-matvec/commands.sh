#!/usr/bin/env bash

set -e

cslc ./layout_matvec.csl --arch wse2 --fabric-dims=9,4 \
--fabric-offsets=4,1 \
--params=width:2,height:2,tile_size:25,iters:1 \
-o out --memcpy --channels=1
cs_python ./run.py --name out --verify
