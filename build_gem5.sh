#!/bin/sh

#sudo apt install build-essential git m4 scons zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev python3-dev libboost-all-dev pkg-config python3-tk

git submodule update --init --recursive gem5
mkdir gem5_bin
cd gem5
python3 `which scons` ../gem5_bin/RISCV/gem5.opt -j
cd ..