#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=4,4 -o out
cs_python run.py --name out
