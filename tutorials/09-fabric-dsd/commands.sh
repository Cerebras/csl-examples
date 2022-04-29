cslc ./code.csl --fabric-dims=3,3 \
--fabric-offsets=1,1 --params=size:64 -o out
cs_python run.py --name out
