#!/usr/bin/env bash
cslc ./code.csl --fabric-dims=3,3 --fabric-offsets=1,1 \
--params=iterations:200 -o out
cs_python run.py --name out --tolerance 0.1
