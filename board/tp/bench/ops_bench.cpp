#include <arm_neon.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include "ggml.h"
#define DIE(...) exit(1)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#include "ops.inc"
static const int L = 16, KVL = 2, D = 64, GROUP = 4, KVR = KVL * D;
template <class F> static double best_of(int reps, F f) { double b = 1e9; for (int r = 0; r < reps; ++r) { double t0 = now(); f(); b = std::min(b, now() - t0); } return b; }
int main(int argc, char ** argv) {
    int ctx_max = argc > 1 ? atoi(argv[1]) : 2048;
    std::vector<float> kcache((size_t) L * ctx_max * KVR), vcache((size_t) L * ctx_max * KVR), q(8 * D), h(8 * D), scores((size_t) 4 * ctx_max);
    for (auto & v : kcache) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : vcache) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : q) v = (rand() % 2000 - 1000) / 1000.0f;
    const float scale = 1.0f / sqrtf((float) D);
    for (int pos : {200, 400}) {
        double t = best_of(5, [&] {
            for (int l = 0; l < L; ++l) for (int kvh = 0; kvh < KVL; ++kvh) {
                // per-head contiguous cache, as in the runtime: [layer][kv head][pos][d]
                const float * kc0 = &kcache[((size_t) l * KVL + kvh) * ctx_max * D];
                const float * vc0 = &vcache[((size_t) l * KVL + kvh) * ctx_max * D];
                attention_heads(4, q.data() + kvh * GROUP * D, kc0, vc0, D, pos, D, scale, scores.data(), ctx_max, h.data() + kvh * GROUP * D);
            }
        });
        printf("ctx_max %d pos %d: runtime attention %7.2f ms per token\n", ctx_max, pos, t * 1e3);
    }
    std::vector<float> x(2048), w(2048, 1.0f), y(4096), act(2048);
    for (auto & v : x) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : y) v = (rand() % 2000 - 1000) / 1000.0f;
    double t;
    t = best_of(20, [&] { for (int i = 0; i < 33; ++i) rmsnorm(h.data(), x.data(), w.data(), 512, 1e-5f); });
    t = best_of(20, [&] { for (int i = 0; i < 33; ++i) rmsnorm(act.data(), x.data(), w.data(), 2048, 1e-5f); });
    printf("rmsnorm x33   %6.3f ms\n", t * 1e3);
    t = best_of(20, [&] { for (int l = 0; l < 16; ++l) silu_mul(act.data(), y.data(), y.data() + 2048, 2048); });
    printf("silu_mul x16  %6.3f ms\n", t * 1e3);
    t = best_of(20, [&] { for (int l = 0; l < 32; ++l) add_into(x.data(), act.data(), 2048); });
    printf("add_into x32  %6.3f ms\n", t * 1e3);
    return 0;
}
