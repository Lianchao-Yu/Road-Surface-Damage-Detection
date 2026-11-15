find_package(benchmark REQUIRED)

add_executable(rd_bench
    bench_smoke.cpp
)

target_link_libraries(rd_bench PRIVATE
    rd_core
    benchmark::benchmark
)

rd_set_warnings(rd_bench)
