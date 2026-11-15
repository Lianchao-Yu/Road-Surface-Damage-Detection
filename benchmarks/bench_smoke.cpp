#include <benchmark/benchmark.h>

#include "road_damage/version.hpp"

namespace {

void BM_VersionString(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(rd::version());
    }
}
BENCHMARK(BM_VersionString);

}  // namespace

BENCHMARK_MAIN();
