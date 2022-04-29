cslc ./code.csl \
--fabric-dims=3,3 --fabric-offsets=1,1 \
--params=num_elements_to_process:2048 \
-o out
cs_python run.py --name out
