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
#include "ops.inc"   // the runtime kernels as committed; "current runtime kernel" below runs them on the interleaved layout

// Variant: PF = prefetch distance in rows (0 = none); probabilities folded into
// the score row (one ld1r per head per row); CH = column chunk (16 or 32).
template <int B, int PF, int CH>
static void attn_v(const float * q, const float * kc, const float * vc, size_t stride, int pos, int d,
                   float scale, float * sc, int sc_stride, float * out) {
    float mx[B];
    #pragma GCC unroll 4
    for (int b = 0; b < B; ++b) mx[b] = -1e30f;
    for (int t = 0; t <= pos; ++t) {
        const float * kt = kc + (size_t) t * stride;
        if (PF) { __builtin_prefetch(kt + PF * stride); __builtin_prefetch(kt + PF * stride + 16); __builtin_prefetch(kt + PF * stride + 32); __builtin_prefetch(kt + PF * stride + 48); }
        float32x4_t acc[B];
        #pragma GCC unroll 4
        for (int b = 0; b < B; ++b) acc[b] = vdupq_n_f32(0);
        for (int i = 0; i < d; i += 4) {
            const float32x4_t k4 = vld1q_f32(kt + i);
            #pragma GCC unroll 4
            for (int b = 0; b < B; ++b) acc[b] = vfmaq_f32(acc[b], k4, vld1q_f32(q + b * d + i));
        }
        #pragma GCC unroll 4
        for (int b = 0; b < B; ++b) { float sv = vaddvq_f32(acc[b]) * scale; sc[b * sc_stride + t] = sv; if (sv > mx[b]) mx[b] = sv; }
    }
    #pragma GCC unroll 4
    for (int b = 0; b < B; ++b) {
        float * s = sc + b * sc_stride;
        float inv = 1.0f / softmax_exp(s, pos + 1, mx[b]);
        const float32x4_t vi = vdupq_n_f32(inv);
        int t = 0;
        for (; t + 4 <= pos + 1; t += 4) vst1q_f32(s + t, vmulq_f32(vld1q_f32(s + t), vi));
        for (; t <= pos; ++t) s[t] *= inv;
    }
    for (int i0 = 0; i0 < d; i0 += CH) {
        float32x4_t acc[B][CH / 4];
        #pragma GCC unroll 4
        for (int b = 0; b < B; ++b) {
            #pragma GCC unroll 8
            for (int j = 0; j < CH / 4; ++j) acc[b][j] = vdupq_n_f32(0);
        }
        for (int t = 0; t <= pos; ++t) {
            const float * vt = vc + (size_t) t * stride + i0;
            if (PF) { __builtin_prefetch(vt + PF * stride); if (CH > 16) __builtin_prefetch(vt + PF * stride + 16); }
            float32x4_t v[CH / 4];
            #pragma GCC unroll 8
            for (int j = 0; j < CH / 4; ++j) v[j] = vld1q_f32(vt + 4 * j);
            #pragma GCC unroll 4
            for (int b = 0; b < B; ++b) {
                const float32x4_t p = vld1q_dup_f32(sc + b * sc_stride + t);
                #pragma GCC unroll 8
                for (int j = 0; j < CH / 4; ++j) acc[b][j] = vfmaq_f32(acc[b][j], v[j], p);
            }
        }
        #pragma GCC unroll 4
        for (int b = 0; b < B; ++b) {
            #pragma GCC unroll 8
            for (int j = 0; j < CH / 4; ++j) vst1q_f32(out + b * d + i0 + 4 * j, acc[b][j]);
        }
    }
}

static const int L = 16, KVL = 2, D = 64, GROUP = 4, KVR = KVL * D;
template <class F> static double best_of(int reps, F f) { double b = 1e9; for (int r = 0; r < reps; ++r) { double t0 = now(); f(); b = std::min(b, now() - t0); } return b; }
int main(int argc, char ** argv) {
    int ctx_max = 2048, pos = argc > 1 ? atoi(argv[1]) : 400;
    std::vector<float> kcache((size_t) L * ctx_max * KVR), vcache((size_t) L * ctx_max * KVR), q(8 * D), h(8 * D), ref(8 * D), scores((size_t) 4 * ctx_max);
    for (auto & v : kcache) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : vcache) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : q) v = (rand() % 2000 - 1000) / 1000.0f;
    const float scale = 1.0f / sqrtf((float) D);
    auto run = [&](auto fn) { for (int l = 0; l < L; ++l) for (int kvh = 0; kvh < KVL; ++kvh) {
        const float * kc0 = &kcache[((size_t) l * ctx_max) * KVR + kvh * D];
        const float * vc0 = &vcache[((size_t) l * ctx_max) * KVR + kvh * D];
        fn(q.data() + kvh * GROUP * D, kc0, vc0, (size_t) KVR, pos, D, scale, scores.data(), ctx_max, h.data() + kvh * GROUP * D); } };
    double t;
    t = best_of(5, [&] { run([](auto... a) { attention_heads(4, a...); }); }); ref = h;
    printf("pos %d  current runtime kernel          %6.2f ms\n", pos, t * 1e3);
    auto diff = [&] { float m = 0; for (int i = 0; i < 8 * D; ++i) m = std::max(m, fabsf(h[i] - ref[i])); return m; };
    t = best_of(5, [&] { run(attn_v<4, 0, 16>); }); printf("pos %d  p folded, ld1r, no prefetch    %6.2f ms  (diff %.2g)\n", pos, t * 1e3, diff());
    t = best_of(5, [&] { run(attn_v<4, 2, 16>); }); printf("pos %d  + prefetch 2 rows ahead        %6.2f ms  (diff %.2g)\n", pos, t * 1e3, diff());
    t = best_of(5, [&] { run(attn_v<4, 4, 16>); }); printf("pos %d  + prefetch 4 rows ahead        %6.2f ms  (diff %.2g)\n", pos, t * 1e3, diff());
    t = best_of(5, [&] { run(attn_v<4, 8, 16>); }); printf("pos %d  + prefetch 8 rows ahead        %6.2f ms  (diff %.2g)\n", pos, t * 1e3, diff());
    t = best_of(5, [&] { run(attn_v<4, 4, 32>); }); printf("pos %d  chunk 32, prefetch 4           %6.2f ms  (diff %.2g)\n", pos, t * 1e3, diff());
    // memory floor: touch every K and V row of the same sweep (256 B per row, 512 B stride)
    float32x4_t acc = vdupq_n_f32(0);
    t = best_of(5, [&] { for (int l = 0; l < L; ++l) for (int kvh = 0; kvh < KVL; ++kvh) for (int pass = 0; pass < 2; ++pass) {
        const float * base = &(pass ? vcache : kcache)[((size_t) l * ctx_max) * KVR + kvh * D];
        for (int tt = 0; tt <= pos; ++tt) { const float * r = base + (size_t) tt * KVR; acc = vaddq_f32(acc, vaddq_f32(vaddq_f32(vld1q_f32(r), vld1q_f32(r + 16)), vaddq_f32(vld1q_f32(r + 32), vld1q_f32(r + 48)))); } } });
    printf("pos %d  memory floor, interleaved layout  %6.2f ms  (%g)\n", pos, t * 1e3, vaddvq_f32(acc));
    // contiguous per-head layout [l][kvh][pos][d]: stride d
    auto runc = [&](auto fn) { for (int l = 0; l < L; ++l) for (int kvh = 0; kvh < KVL; ++kvh) {
        const float * kc0 = &kcache[((size_t) l * KVL + kvh) * ctx_max * D];
        const float * vc0 = &vcache[((size_t) l * KVL + kvh) * ctx_max * D];
        fn(q.data() + kvh * GROUP * D, kc0, vc0, (size_t) D, pos, D, scale, scores.data(), ctx_max, h.data() + kvh * GROUP * D); } };
    t = best_of(5, [&] { runc(attn_v<4, 0, 16>); }); printf("pos %d  contiguous layout, no prefetch  %6.2f ms\n", pos, t * 1e3);
    t = best_of(5, [&] { runc(attn_v<4, 4, 16>); }); printf("pos %d  contiguous, prefetch 4          %6.2f ms\n", pos, t * 1e3);
    t = best_of(5, [&] { runc(attn_v<4, 4, 32>); }); printf("pos %d  contiguous, chunk 32, prefetch 4 %6.2f ms\n", pos, t * 1e3);
    t = best_of(5, [&] { runc(attn_v<4, 0, 32>); }); printf("pos %d  contiguous, chunk 32, no prefetch %6.2f ms\n", pos, t * 1e3);
    t = best_of(5, [&] { for (int l = 0; l < L; ++l) for (int kvh = 0; kvh < KVL; ++kvh) for (int pass = 0; pass < 2; ++pass) {
        const float * base = &(pass ? vcache : kcache)[((size_t) l * KVL + kvh) * ctx_max * D];
        for (int tt = 0; tt <= pos; ++tt) { const float * r = base + (size_t) tt * D; acc = vaddq_f32(acc, vaddq_f32(vaddq_f32(vld1q_f32(r), vld1q_f32(r + 16)), vaddq_f32(vld1q_f32(r + 32), vld1q_f32(r + 48)))); } } });
    printf("pos %d  memory floor, contiguous layout   %6.2f ms  (%g)\n", pos, t * 1e3, vaddvq_f32(acc));
    return 0;
}
