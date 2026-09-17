// The per-token CPU work that does not grow with position, timed as in tp_llama.cpp.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include "ggml.h"
#include "ggml-quants.h"
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void rmsnorm(float * out, const float * x, const float * w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    float scale = 1.0f / sqrtf((float) (ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}
static void rope(float * x, int n_heads, int d, int pos, float base, const float * freqs) {
    for (int i = 0; i < d / 2; ++i) {
        float theta = pos * powf(base, -2.0f * i / d);
        if (freqs) theta /= freqs[i];
        float c = cosf(theta), s = sinf(theta);
        for (int h = 0; h < n_heads; ++h) {
            float * p = x + h * d + 2 * i;
            float x0 = p[0], x1 = p[1];
            p[0] = x0 * c - x1 * s;
            p[1] = x0 * s + x1 * c;
        }
    }
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
template <class F> static double best_of(int reps, F f) { double b = 1e9; for (int r = 0; r < reps; ++r) { double t0 = now(); f(); b = std::min(b, now() - t0); } return b; }
int main() {
    const int E = 2048, FF = 2048, V = 32064;
    std::vector<float> x(E), h(E), w(E, 1.0f), y(2 * FF), act(FF), logits(V), freqs(32, 1.0f), qkv(12 * 64);
    for (auto & v : x) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : y) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : logits) v = (rand() % 2000 - 1000) / 1000.0f;
    std::vector<block_q8_0> q8(256);
    double t;
    t = best_of(20, [&] { for (int i = 0; i < 33; ++i) rmsnorm(h.data(), x.data(), w.data(), E, 1e-5f); });
    printf("rmsnorm x33            %6.2f ms\n", t * 1e3);
    t = best_of(20, [&] { for (int l = 0; l < 16; ++l) { rope(qkv.data(), 8, 64, 300, 500000.0f, freqs.data()); rope(qkv.data() + 512, 2, 64, 300, 500000.0f, freqs.data()); } });
    printf("rope x32               %6.2f ms\n", t * 1e3);
    t = best_of(20, [&] { for (int l = 0; l < 16; ++l) for (int i = 0; i < FF; ++i) act[i] = silu(y[i]) * y[FF + i]; });
    printf("silu*up x16            %6.2f ms\n", t * 1e3);
    t = best_of(20, [&] { int best = 0; for (int i = 1; i < V; ++i) if (logits[i] > logits[best]) best = i; x[0] += best; });
    printf("argmax 32064           %6.2f ms\n", t * 1e3);
    t = best_of(20, [&] { for (int l = 0; l < 32; ++l) for (int i = 0; i < E; ++i) x[i] += h[i]; });
    printf("residual add x32       %6.2f ms\n", t * 1e3);
    t = best_of(20, [&] { for (int c = 0; c < 65; ++c) quantize_row_q8_0_ref(x.data(), q8.data(), E); });
    printf("quantize_row_q8_0_ref x65 (in the fabric bucket) %6.2f ms\n", t * 1e3);
    t = best_of(20, [&] { for (int c = 0; c < 65; ++c) { quantize_row_q8_0_ref(x.data(), q8.data(), E); uint8_t xb[8192 + 1024]; float * xd = (float *) (xb + E); for (int b = 0; b < E / 32; ++b) { memcpy(xb + 32 * b, q8[b].qs, 32); xd[b] = ggml_fp16_to_fp32(q8[b].d); } x[1] += xd[0]; } });
    printf("quantize + pack x65    (in the fabric bucket) %6.2f ms\n", t * 1e3);
    return 0;
}
