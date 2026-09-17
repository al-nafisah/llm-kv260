// C testbench for gemv2: lay the contract's vectors out as the runtime will
// (per port, row pairs with interleaved chunks), run fetch then mac through
// C++ streams, compare y byte for byte. Then the drain command on FIFOs
// that still hold beats.
#include "gemv2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<unsigned char> slurp(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
    fclose(f);
    return buf;
}

// Byte offset of chunk c of row r inside the engine layout of a [rows, k]
// matrix: port p = r / R holds rows [pR, pR+R); within it row pR+i pairs
// with row pR+H+i (H = R/2), chunk by chunk. Same formula as the runtime.
static size_t chunk_offset(int r, int c, int rows, int k) {
    const int R = rows / PORTS, H = R / 2, nch = k / 256;
    const int p = r / R, i = (r % R) % H, which = (r % R) / H;
    return ((size_t)p * R * nch + ((size_t)i * nch + c) * 2 + which) * CHUNK_BYTES;
}

static int run_case(const std::string &dir) {
    int k = 0, rows = 0;
    FILE *m = fopen((dir + "/meta.txt").c_str(), "r");
    if (!m || fscanf(m, "%d %d", &k, &rows) != 2) { fprintf(stderr, "bad meta in %s\n", dir.c_str()); return 1; }
    fclose(m);
    std::vector<unsigned char> w = slurp(dir + "/w.bin"), x = slurp(dir + "/x.bin"), y_ref = slurp(dir + "/y.bin");
    const int row_bytes = k / QK * BLOCK_BYTES, nch = k / 256;
    const int R = rows / PORTS, H = R / 2;
    if (rows % (MACS * 4) || w.size() != (size_t)rows * row_bytes) { fprintf(stderr, "%s: unusable shape\n", dir.c_str()); return 1; }
    x.resize((x.size() + 15) / 16 * 16, 0);

    std::vector<unsigned char> lay(w.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < nch; ++c)
            memcpy(&lay[chunk_offset(r, c, rows, k)], &w[(size_t)r * row_bytes + (size_t)c * CHUNK_BYTES], CHUNK_BYTES);

    std::vector<unsigned char> y(rows * 4, 0xAA);
    const beat_t *wb = reinterpret_cast<const beat_t *>(lay.data());
    beat_t *yb = reinterpret_cast<beat_t *>(y.data());
    const size_t port_beats = (size_t)R * row_bytes / 16, mac_ybeats = (size_t)H * 4 / 16;
    hls::stream<beat_t> s[MACS];
    gemv_fetch(wb, wb + port_beats, wb + 2 * port_beats, wb + 3 * port_beats, (int)port_beats,
               s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
    gemv_mac(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], reinterpret_cast<const beat_t *>(x.data()),
             yb, yb + mac_ybeats, yb + 2 * mac_ybeats, yb + 3 * mac_ybeats,
             yb + 4 * mac_ybeats, yb + 5 * mac_ybeats, yb + 6 * mac_ybeats, yb + 7 * mac_ybeats, k, H);
    int bad = 0;
    for (int r = 0; r < rows; ++r) {
        if (memcmp(&y[4 * r], &y_ref[4 * r], 4) != 0) {
            float got, want;
            memcpy(&got, &y[4 * r], 4); memcpy(&want, &y_ref[4 * r], 4);
            if (bad < 8) printf("  row %d: got %.9g want %.9g\n", r, got, want);
            bad++;
        }
    }
    for (int i = 0; i < MACS; ++i) if (!s[i].empty()) { printf("  FIFO %d not empty after the call\n", i); bad++; }
    printf("%s: k=%d rows=%d -> %s (%d/%d rows differ)\n", dir.c_str(), k, rows, bad ? "MISMATCH" : "bit-exact", bad, rows);
    return bad ? 1 : 0;
}

static int drain_test() {
    hls::stream<beat_t> s[MACS];
    beat_t dummy[16];
    for (int i = 0; i < MACS; ++i) for (int j = 0; j <= i; ++j) s[i].write(beat_t(j));
    gemv_mac(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, 256, 0);
    int bad = 0;
    for (int i = 0; i < MACS; ++i) if (!s[i].empty()) bad++;
    printf("drain: %s\n", bad ? "FIFOs still hold beats" : "all FIFOs empty");
    return bad ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "../int4_core/vectors";
    const char *cases[] = {"k256_n32", "k512_n64", "k1024_n32", "k2048_n64", "k8192_n32"};
    int failures = 0;
    for (const char *c : cases) failures += run_case(std::string(root) + "/" + c);
    failures += drain_test();
    printf(failures ? "FAIL\n" : "PASS\n");
    return failures;
}
