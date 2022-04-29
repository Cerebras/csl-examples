cslc ./code.csl --fabric-dims=12,12 --fabric-offsets=1,1 \
-o out --params=size:10,zDim:10 --colors=tallyOut:8,tscColor:9,iterColor:10
cs_python run.py --name out --size 10 --zDim 10 --iterations 10 \
--out_color 8 --tsc_color 9 --iter_color 10
