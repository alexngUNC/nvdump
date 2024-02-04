### Description
Tool for printing out information from NVIDIA GPUs. It is based on accessing the QMD/TMD structs, as done by Joshua Bakita as a tiny component in his SM partitioning work (https://www.cs.unc.edu/~jbakita/rtas23-ae/).

### Compilation: 
/usr/local/cuda/bin/nvcc start.cu -lcuda -I/usr/local/cuda/include -ldl -o start
/usr/local/cuda/bin/nvcc vecAdd.cu -lcuda -I/usr/local/cuda/include -ldl -o vecAdd
