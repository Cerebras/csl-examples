#!/usr/bin/env bash
cslc ./code.csl --params=problemDepth:3,problemHeight:10,problemWidth:10,computeHeight:5,computeWidth:5,timeSteps:4,ghostCells:2 --fabric-dims=7,7 --fabric-offsets=1,1 -o out
cs_python run.py --name out
