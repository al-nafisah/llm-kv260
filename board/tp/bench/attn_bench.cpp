// Attention microbenchmark: the exact loop from tp_llama.cpp forward() vs alternatives.
// Layout matches the runtime: kcache/vcache [layer][pos][kv_rows], kv_rows = 2 kv heads * 64.
#include <arm_neon.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const int L = 16, HL = 8, KVL = 2, D = 64, GROUP = 4, KVR = KVL * D;
static int ctx_max = 2048;
static std::vector<float> kcache, vcache, scores, h, q;

__attribute__((noinline)) static void attn_ref(int l, int pos) {
    const float scale = 1.0f / sqrtf((float) D);
    for (int hq = 0; hq < HL; ++hq) {
        const int kvh = hq / GROUP;
        const float * qh = q.data() + hq * D;
        float mx = -1e30f;
        for (int t = 0; t <= pos; ++t) {
            const float * kt = &kcache[((size_t) l * ctx_max + t) * KVR + kvh * D];
            float s = 0;
            for (int i = 0; i < D; ++i) s += qh[i] * kt[i];
            s *= scale; scores[t] = s; if (s > mx) mx = s;
        }
        float sum = 0;
        for (int t = 0; t <= pos; ++t) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
        float inv = 1.0f / sum;
        float * o = h.data() + hq * D;
        for (int i = 0; i < D; ++i) o[i] = 0;
        for (int t = 0; t <= pos; ++t) {
            const float * vt = &vcache[((size_t) l * ctx_max + t) * KVR + kvh * D];
            float p = scores[t] * inv;
            for (int i = 0; i < D; ++i) o[i] += p * vt[i];
        }
    }
}

// NEON, GQA-batched: one pass over K serves the 4 query heads of the group; same for V.
// scores4: [4][ctx]
static std::vector<float> scores4;
static inline float hsum(float32x4_t v) { return vaddvq_f32(v); }
__attribute__((noinline)) static void attn_neon_kvh(int l, int pos, int kvh, float * scores_base) {
    const float scale = 1.0f / sqrtf((float) D);
    const float * q0 = q.data() + (kvh * GROUP) * D;
    float * sc[4]; for (int g = 0; g < 4; ++g) sc[g] = scores_base + g * ctx_max;
    float mx[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    for (int t = 0; t <= pos; ++t) {
        const float * kt = &kcache[((size_t) l * ctx_max + t) * KVR + kvh * D];
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int i = 0; i < D; i += 4) {
            float32x4_t k4 = vld1q_f32(kt + i);
            a0 = vfmaq_f32(a0, k4, vld1q_f32(q0 + 0 * D + i));
            a1 = vfmaq_f32(a1, k4, vld1q_f32(q0 + 1 * D + i));
            a2 = vfmaq_f32(a2, k4, vld1q_f32(q0 + 2 * D + i));
            a3 = vfmaq_f32(a3, k4, vld1q_f32(q0 + 3 * D + i));
        }
        float s0 = hsum(a0) * scale, s1 = hsum(a1) * scale, s2 = hsum(a2) * scale, s3 = hsum(a3) * scale;
        sc[0][t] = s0; sc[1][t] = s1; sc[2][t] = s2; sc[3][t] = s3;
        if (s0 > mx[0]) mx[0] = s0; if (s1 > mx[1]) mx[1] = s1; if (s2 > mx[2]) mx[2] = s2; if (s3 > mx[3]) mx[3] = s3;
    }
    float inv[4];
    for (int g = 0; g < 4; ++g) {
        float sum = 0;
        for (int t = 0; t <= pos; ++t) { sc[g][t] = expf(sc[g][t] - mx[g]); sum += sc[g][t]; }
        inv[g] = 1.0f / sum;
    }
    float32x4_t o[4][D / 4];
    for (int g = 0; g < 4; ++g) for (int i = 0; i < D / 4; ++i) o[g][i] = vdupq_n_f32(0);
    for (int t = 0; t <= pos; ++t) {
        const float * vt = &vcache[((size_t) l * ctx_max + t) * KVR + kvh * D];
        float32x4_t p0 = vdupq_n_f32(sc[0][t] * inv[0]), p1 = vdupq_n_f32(sc[1][t] * inv[1]);
        float32x4_t p2 = vdupq_n_f32(sc[2][t] * inv[2]), p3 = vdupq_n_f32(sc[3][t] * inv[3]);
        for (int i = 0; i < D / 4; ++i) {
            float32x4_t v4 = vld1q_f32(vt + 4 * i);
            o[0][i] = vfmaq_f32(o[0][i], v4, p0);
            o[1][i] = vfmaq_f32(o[1][i], v4, p1);
            o[2][i] = vfmaq_f32(o[2][i], v4, p2);
            o[3][i] = vfmaq_f32(o[3][i], v4, p3);
        }
    }
    for (int g = 0; g < 4; ++g) for (int i = 0; i < D / 4; ++i) vst1q_f32(h.data() + (kvh * GROUP + g) * D + 4 * i, o[g][i]);
}
static void attn_neon(int l, int pos) { for (int kvh = 0; kvh < KVL; ++kvh) attn_neon_kvh(l, pos, kvh, scores4.data() + kvh * 4 * ctx_max); }

// Threaded: KVL=2 kv heads -> 2 workers (one per kv head), spin barrier.
struct pool {
    std::atomic<int> gen{0}, done{0};
    int l = 0, pos = 0;
    std::vector<std::thread> th;
    bool stop = false;
    void start(int n) {
        for (int w = 1; w < n; ++w) th.emplace_back([this, w] {
            int seen = 0;
            for (;;) {
                while (gen.load(std::memory_order_acquire) == seen) { if (stop) return; }
                seen = gen.load(std::memory_order_acquire);
                attn_neon_kvh(l, pos, w, scores4.data() + w * 4 * ctx_max);
                done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    void run(int ll, int p) {
        l = ll; pos = p; done.store(0);
        gen.fetch_add(1, std::memory_order_release);
        attn_neon_kvh(l, pos, 0, scores4.data());
        while (done.load(std::memory_order_acquire) < (int) th.size()) {}
    }
    void join() { stop = true; for (auto & t : th) t.join(); }
};

int main(int argc, char ** argv) {
    int pos = argc > 1 ? atoi(argv[1]) : 400;
    kcache.assign((size_t) L * ctx_max * KVR, 0); vcache.assign((size_t) L * ctx_max * KVR, 0);
    for (auto & v : kcache) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : vcache) v = (rand() % 2000 - 1000) / 1000.0f;
    q.resize(HL * D); for (auto & v : q) v = (rand() % 2000 - 1000) / 1000.0f;
    h.resize(HL * D); scores.resize(ctx_max); scores4.resize(KVL * 4 * ctx_max);
    const int reps = 5;
    std::vector<float> ref;
    double best;
    // reference
    best = 1e9;
    for (int r = 0; r < reps; ++r) { double t0 = now(); for (int l = 0; l < L; ++l) attn_ref(l, pos); best = std::min(best, now() - t0); }
    ref = h;
    printf("pos %d: reference scalar loop      %7.2f ms per token (16 layers, 8 heads)\n", pos, best * 1e3);
    best = 1e9;
    for (int r = 0; r < reps; ++r) { double t0 = now(); for (int l = 0; l < L; ++l) attn_neon(l, pos); best = std::min(best, now() - t0); }
    double maxdiff = 0; for (size_t i = 0; i < h.size(); ++i) maxdiff = std::max(maxdiff, (double) fabsf(h[i] - ref[i]));
    printf("pos %d: neon, 4 heads per K/V pass  %7.2f ms per token (max |diff| vs reference %.3g)\n", pos, best * 1e3, maxdiff);
    pool P; P.start(KVL);
    best = 1e9;
    for (int r = 0; r < reps; ++r) { double t0 = now(); for (int l = 0; l < L; ++l) P.run(l, pos); best = std::min(best, now() - t0); }
    maxdiff = 0; for (size_t i = 0; i < h.size(); ++i) maxdiff = std::max(maxdiff, (double) fabsf(h[i] - ref[i]));
    printf("pos %d: neon + 2 threads            %7.2f ms per token (max |diff| vs reference %.3g)\n", pos, best * 1e3, maxdiff);
    P.join();
    return 0;
}
