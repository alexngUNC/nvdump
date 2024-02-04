### Description
Tool for printing out information from NVIDIA GPUs. It is based on accessing the QMD/TMD structs, as done by Joshua Bakita as a tiny component in his SM partitioning work (https://www.cs.unc.edu/~jbakita/rtas23-ae/).

Reference the QMD/TMD struct chart found here: https://nvidia.github.io/open-gpu-doc/classes/compute/clc7c0qmd.h

Supported by CUDA 11+

### Compilation: 
#### Start program
/usr/local/cuda/bin/nvcc start.cu -lcuda -I/usr/local/cuda/include -ldl -o start

#### Demo vector addition program
/usr/local/cuda/bin/nvcc vecAdd.cu -lcuda -I/usr/local/cuda/include -ldl -o vecAdd
