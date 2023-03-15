#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=10,10 -o out
cs_python run.py --name out
