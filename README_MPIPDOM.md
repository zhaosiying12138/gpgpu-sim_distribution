This is an implementation, test and log record for Multi-Path IPDOM Machenism By zhaosiying12138.

Normal Usage:
nvcc --cudart shared -arch=sm_30 test_mpipdom_dualpath1.cu -O0 -Xcicc -O0 -Xptxas -O0 -g
nvcc --cudart shared -arch=sm_30 test_mpipdom_dualpath2.cu -O0 -Xcicc -O0 -Xptxas -O0 -g
PS: It seems dualpath will be easier than nested branch, so it will be better as a testcase
at my early development stage.

1. Test Split Table and Reconvergence Table Machenism
nvcc --cudart shared -arch=sm_30 test_mpipdom_st_rt.cu -O0 -Xcicc -O0 -Xptxas -O0 -g
The record is in test_mpipdom_st_rt.log

2. Test Performance Varying maximum Warp Splits parallelization
Use test_mpipdom_nested_branch.cu as host code and test_mpipdom_nested_branch.ptx as device kernel ptx,
then run compile_ptx.sh to merge and compile them together into a fatbinary.
You need modify gpgpusim.config to set #SP Units and modify shader.cc to disable SIMD pipeline
as my blog illustrates.
The record is in test_mpipdom_nested_branch_baseline.log and test_mpipdom_nested_branch_sp1/2/4.log

3. Test Scoreboard Machenism
Use test_mpipdom_scoreboard.cu as host code and test_mpipdom_scoreboard.ptx as device kernel ptx,
then run compile_ptx.sh to merge and compile them together into a fatbinary.
The record is in test_mpipdom_scoreboard.log
