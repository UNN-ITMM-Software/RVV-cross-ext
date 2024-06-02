#!/bin/sh

#sudo apt install build-essential git m4 scons zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev python3-dev libboost-all-dev pkg-config python3-tk

if [ $# -eq 0 ];then
    th="-j1"
else
    th="$@"
fi

git submodule update --init --recursive gem5
cd gem5
pip install -r requirements.txt
python3 `which scons` ./build/RISCV/gem5.opt $th
cd ..
