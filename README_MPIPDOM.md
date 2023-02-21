This is an implementation of Multi-Path IPDOM By zhaosiying12138.

Usage:
nvcc --cudart shared -arch=sm_30 test_mpipdom_dualpath1.cu -O0 -Xcicc -O0 -Xptxas -O0 -g
nvcc --cudart shared -arch=sm_30 test_mpipdom_dualpath2.cu -O0 -Xcicc -O0 -Xptxas -O0 -g
