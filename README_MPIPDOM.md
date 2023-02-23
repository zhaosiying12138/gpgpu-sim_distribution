This is an implementation of Multi-Path IPDOM By zhaosiying12138.

Usage:
nvcc --cudart shared -arch=sm_30 test_mpipdom_dualpath1.cu -O0 -Xcicc -O0 -Xptxas -O0 -g
nvcc --cudart shared -arch=sm_30 test_mpipdom_dualpath2.cu -O0 -Xcicc -O0 -Xptxas -O0 -g

1. Testing Split Table and Reconvergence Table
xxxxx

2. Testing Scoreboard
Use test_mpipdom_scoreboard.cu as host code and test_mpipdom_scoreboard.ptx as device kernel ptx,
then run compile_ptx.sh to merge and compile them together into a fatbinary.
The log is in test_mpipdom_scoreboard.log

