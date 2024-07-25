CC = gcc
NVCC ?= nvcc
CFLAGS = -fPIC
LDFLAGS = -lcuda -I/usr/local/cuda/include -ldl

.PHONY: clean

all: start vecAdd l1

start: start.cu
	$(NVCC) start.cu $(LDFLAGS) -o bin/start	

vecAdd: vecAdd.cu
	$(NVCC) vecAdd.cu $(LDFLAGS) -o bin/vecAdd

l1: l1.cu
	$(NVCC) l1.cu $(LDFLAGS) -o bin/l1 -g

l1prompt: l1prompt.cu
	$(NVCC) l1prompt.cu $(LDFLAGS) -o bin/l1prompt -g

l1args: l1args.cu
	$(NVCC) l1args.cu $(LDFLAGS) -o bin/l1args -g

clean:
	rm -rf bin/*
