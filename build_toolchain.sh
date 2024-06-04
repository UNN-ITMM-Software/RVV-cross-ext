#!/bin/sh

#sudo apt-get install autoconf automake autotools-dev curl python3 python3-pip libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev libslirp-dev

if [ $# -eq 0 ];then
    th="-j1"
else
    th="$@"
fi

git submodule update --init riscv-gnu-toolchain
mkdir -p toolchain
cd riscv-gnu-toolchain
git submodule update --init --recursive binutils gcc gdb
cd binutils
git apply ../binutils.patch
cd ../gcc
git apply ../gcc.patch
cd ../gdb
git apply ../gdb.patch
cd ..
pwd_path=`pwd`
./configure --prefix=${pwd_path}/../toolchain/ --disable-linux --with-arch=rv64gcv
make newlib $th
cd ..