#!/usr/bin/env bash
cslc ./code.csl --fabric-dims=3,4 --fabric-offsets=1,1 -o out
cs_python run.py --name out
