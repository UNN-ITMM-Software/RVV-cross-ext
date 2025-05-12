set(CMAKE_SYSTEM_NAME elf)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

SET(CMAKE_CROSSCOMPILING 1)

set(CMAKE_CXX_COMPILER riscv64-unknown-elf-g++)
set(CMAKE_C_COMPILER   riscv64-unknown-elf-gcc)

set(CMAKE_CXX_FLAGS_INIT " -O2 -march=rv64gcv -mabi=lp64d -static -ffast-math")
set(CMAKE_C_FLAGS_INIT   " -O2 -march=rv64gcv -mabi=lp64d -ffast-math")

SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
