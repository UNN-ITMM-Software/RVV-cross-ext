include_directories(
    ../../../pica/include
    ../../)

add_executable       (pusher-baseline Main.cpp)
target_link_libraries(pusher-baseline benchmark-utility)


add_executable       (pusher-baseline-unroll Main-unroll.cpp)
target_link_libraries(pusher-baseline-unroll benchmark-utility)


add_executable       (pusher-vectorized-rvv Main-rvv.cpp)
target_link_libraries(pusher-vectorized-rvv benchmark-utility)

add_executable       (pusher-vectorized-rvv-unroll Main-rvv-unroll.cpp)
target_link_libraries(pusher-vectorized-rvv-unroll benchmark-utility)

add_executable       (pusher-vectorized-rvv-with-cross Main-rvv-cross.cpp)
target_link_libraries(pusher-vectorized-rvv-with-cross benchmark-utility)

add_executable       (pusher-vectorized-rvv-with-cross-unroll Main-rvv-cross-unroll.cpp)
target_link_libraries(pusher-vectorized-rvv-with-cross-unroll benchmark-utility)
