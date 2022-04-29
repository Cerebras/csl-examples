cslc ./code.csl --fabric-dims=6,6 --fabric-offsets=1,1 \
--colors=x_in:1,b_in:2,y_out:3,Ax_out:4,sentinel:43 -o out
cs_python run.py --name out
