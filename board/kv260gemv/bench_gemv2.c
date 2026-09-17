// gemv2 on the board: exactness against the contract's vectors (the n32
// cases), then bandwidth on one rank's shard shapes, cold (fetch and MAC
// started together) and prefetched (fetch started 1 ms earlier), then a
// token's 65 calls with the fetch of each matrix issued right after the
// MAC of the previous one, with and without a 300 us gap between calls
// standing in for the CPU and the exchange.
//
//   ./bench_gemv2 ~/vectors [udmabuf0 [case ...]]   # e.g. k512_n2048 k2048_n768 from gen_vectors.py DIR 512,2048 2048,768
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "gemv2_dev.h"
#include "udmabuf.h"

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static size_t slurp(const char *path, void *dst, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    size_t n = fread(dst, 1, cap, f);
    fclose(f);
    return n;
}

// Four port regions, 64 KB-aligned, ports 1 and 3 offset by 16 KB.
typedef struct { uint8_t *q[4]; uint64_t phys[4]; int rows, k; } mat_t;
static mat_t new_mat(udmabuf_t *mem, int rows, int k) {
    mat_t m = {.rows = rows, .k = k};
    size_t quarter = (size_t)rows / 4 * k / 32 * 18;
    for (int p = 0; p < 4; ++p) {
        size_t off = (p & 1) * (16 << 10);
        uint8_t *b = udmabuf_alloc(mem, quarter + off, 64 << 10);
        if (!b) { fprintf(stderr, "out of udmabuf memory\n"); exit(2); }
        m.q[p] = b + off; m.phys[p] = udmabuf_phys(mem, b) + off;
    }
    return m;
}
static void put_row(mat_t *m, int r, const uint8_t *src) {   // src: one Q4_0 row of k/32*18 bytes
    for (int c = 0; c < m->k / 256; ++c) { int p; size_t o = gemv2_chunk_offset(r, c, m->rows, m->k, &p); memcpy(m->q[p] + o, src + (size_t)c * 144, 144); }
}
static void clean_mat(mat_t *m) { size_t quarter = (size_t)m->rows / 4 * m->k / 32 * 18; for (int p = 0; p < 4; ++p) udmabuf_clean(m->q[p], quarter); }

static int check_case(gemv2_dev_t *dev, udmabuf_t *mem, const char *root, const char *name) {
    char path[512];
    int k, rows;
    snprintf(path, sizeof path, "%s/%s/meta.txt", root, name);
    FILE *f = fopen(path, "r");
    if (!f || fscanf(f, "%d %d", &k, &rows) != 2) { fprintf(stderr, "bad %s\n", path); return 1; }
    fclose(f);
    size_t rb = (size_t)k / 32 * 18, xbytes = k + k / 8, ybytes = (size_t)rows * 4;
    uint8_t *w = malloc(rows * rb), *y_ref = malloc(ybytes);
    uint8_t *x = udmabuf_alloc(mem, xbytes + 64, 4096), *y = udmabuf_alloc(mem, ybytes, 4096);
    snprintf(path, sizeof path, "%s/%s/w.bin", root, name); slurp(path, w, rows * rb);
    snprintf(path, sizeof path, "%s/%s/x.bin", root, name); slurp(path, x, xbytes);
    snprintf(path, sizeof path, "%s/%s/y.bin", root, name); slurp(path, y_ref, ybytes);
    mat_t m = new_mat(mem, rows, k);
    for (int r = 0; r < rows; ++r) put_row(&m, r, w + r * rb);
    clean_mat(&m);
    memset(y, 0xAA, ybytes);
    udmabuf_clean(x, xbytes); udmabuf_clean(y, ybytes);
    gemv2_fetch_start(dev, m.phys, k, rows);
    gemv2_mac_start(dev, udmabuf_phys(mem, x), udmabuf_phys(mem, y), k, rows);
    gemv2_mac_wait(dev);
    udmabuf_invalidate(y, ybytes);
    int bad = 0;
    for (int r = 0; r < rows; ++r) if (memcmp(y + 4 * r, y_ref + 4 * r, 4)) { if (bad < 5) { float g, wnt; memcpy(&g, y + 4 * r, 4); memcpy(&wnt, y_ref + 4 * r, 4); printf("  row %d: got %.9g want %.9g\n", r, g, wnt); } bad++; }
    printf("%-12s k=%-5d rows=%-4d %s (%d/%d rows differ)\n", name, k, rows, bad ? "MISMATCH" : "bit-exact", bad, rows);
    free(w); free(y_ref);
    return bad != 0;
}

typedef struct { const char *name; int k, rows; } shape_t;

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "vectors";
    const char *pool = argc > 2 ? argv[2] : "udmabuf0";
    gemv2_dev_t dev; udmabuf_t mem;
    if (gemv2_dev_open(&dev)) { fprintf(stderr, "gemv2 overlay not loaded\n"); return 2; }
    if (udmabuf_open(&mem, pool)) return 2;
    if (!gemv2_mac_idle(&dev)) { fprintf(stderr, "MAC kernel busy: reload the overlay\n"); return 2; }
    double t0 = now(); gemv2_drain(&dev); printf("drain: %.1f us\n", (now() - t0) * 1e6);

    int failures = 0;
    const char *cases[] = {"k256_n32", "k512_n64", "k1024_n32", "k2048_n64", "k8192_n32"};
    if (argc > 3) for (int i = 3; i < argc; ++i) failures += check_case(&dev, &mem, root, argv[i]);   // extra cases by name
    else for (int i = 0; i < 5; ++i) failures += check_case(&dev, &mem, root, cases[i]);
    printf(failures ? "EXACTNESS: FAIL\n" : "EXACTNESS: PASS\n");

    // One rank's shard shapes (four ranks): random bytes, value-independent timing.
    mem.used = 0;
    shape_t S[] = {{"qkv", 2048, 768}, {"o", 512, 2048}, {"gateup", 2048, 4096}, {"down", 2048, 2048}, {"lm_head", 2048, 32064}};
    mat_t m[5];
    srand(1);
    for (int i = 0; i < 5; ++i) {
        m[i] = new_mat(&mem, S[i].rows, S[i].k);
        size_t quarter = (size_t)S[i].rows / 4 * S[i].k / 32 * 18;
        for (int p = 0; p < 4; ++p) for (size_t b = 0; b < quarter; ++b) m[i].q[p][b] = rand();
        clean_mat(&m[i]);
    }
    uint8_t *x = udmabuf_alloc(&mem, 8192 + 1024 + 64, 4096), *y = udmabuf_alloc(&mem, 32064 * 4, 4096);
    for (int i = 0; i < 8192 + 1024; ++i) x[i] = rand();
    udmabuf_clean(x, 8192 + 1024);
    uint64_t xp = udmabuf_phys(&mem, x), yp = udmabuf_phys(&mem, y);

    printf("\n%-8s %8s | %10s %8s | %10s %8s\n", "shape", "MB", "cold", "GB/s", "prefetched", "GB/s");
    for (int i = 0; i < 5; ++i) {
        size_t wb = (size_t)S[i].rows * S[i].k / 32 * 18;
        int reps = wb > 10e6 ? 20 : 200;
        double cold = 1e9, pre = 1e9;
        for (int r = 0; r < reps; ++r) {
            double a = now();
            gemv2_fetch_start(&dev, m[i].phys, S[i].k, S[i].rows);
            gemv2_mac_start(&dev, xp, yp, S[i].k, S[i].rows);
            gemv2_mac_wait(&dev);
            double dt = now() - a; if (dt < cold) cold = dt;
        }
        for (int r = 0; r < reps; ++r) {
            gemv2_fetch_start(&dev, m[i].phys, S[i].k, S[i].rows);
            usleep(1000);
            double a = now();
            gemv2_mac_start(&dev, xp, yp, S[i].k, S[i].rows);
            gemv2_mac_wait(&dev);
            double dt = now() - a; if (dt < pre) pre = dt;
        }
        printf("%-8s %8.3f | %7.1f us %8.2f | %7.1f us %8.2f\n", S[i].name, wb / 1e6, cold * 1e6, wb / cold / 1e9, pre * 1e6, wb / pre / 1e9);
    }

    // A token: 16 x (qkv, o, gateup, down) + lm_head, fetch of the next issued after each MAC.
    for (int gap_us = 0; gap_us <= 300; gap_us += 300) {
        int seq[65]; int n = 0;
        for (int l = 0; l < 16; ++l) { seq[n++] = 0; seq[n++] = 1; seq[n++] = 2; seq[n++] = 3; }
        seq[n++] = 4;
        double mac_total = 0, wall0 = now();
        gemv2_fetch_start(&dev, m[seq[0]].phys, S[seq[0]].k, S[seq[0]].rows);
        for (int c = 0; c < 65; ++c) {
            int i = seq[c];
            double a = now();
            gemv2_mac_start(&dev, xp, yp, S[i].k, S[i].rows);
            gemv2_mac_wait(&dev);
            mac_total += now() - a;
            int nx = seq[(c + 1) % 65];
            gemv2_fetch_start(&dev, m[nx].phys, S[nx].k, S[nx].rows);
            if (gap_us) usleep(gap_us);
        }
        double wall = now() - wall0;
        printf("\n65 calls, %d us gap after each: MAC time %.2f ms (%.2f GB/s over 173.8 MB), wall %.2f ms\n", gap_us, mac_total * 1e3, 173.78e6 / mac_total / 1e9, wall * 1e3);
        gemv2_drain(&dev);
    }
    return failures;
}
