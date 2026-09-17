// gemv_mac: see gemv2.h. Eight copies of int4_core's unpack + mac, each fed
// by one FIFO instead of a DRAM pointer. The activation vector is loaded
// once per call into eight private BRAM copies.
#include "gemv2.h"
#include "../int4_core/gemv_core.h"

static void engine(hls::stream<beat_t> &w, const xblock_t xq[MAX_NB], const float xd[MAX_NB],
                   int k, int rows, beat_t *y) {
#pragma HLS DATAFLOW
    hls::stream<block_t> blocks("blocks");
#pragma HLS STREAM variable = blocks depth = 64
    const int nb = k / QK;
    const int beats = rows * (k * 9 / 256);
    unpack_stream(w, beats, blocks);
    mac(blocks, xq, xd, nb, rows, y);
}

// All eight FIFOs in one loop, never one after another: a reader of the
// fetch kernel that a dead process left stalled alternates chunks between
// its port's two FIFOs, so draining them in turn lets it block on the one
// not being read, the count of idle cycles runs out early, and every later
// call reads beats offset by the stale ones (measured: half the rows wrong
// until the next clean drain, buildlog 0012).
static void drain(hls::stream<beat_t> &s0a, hls::stream<beat_t> &s0b,
                  hls::stream<beat_t> &s1a, hls::stream<beat_t> &s1b,
                  hls::stream<beat_t> &s2a, hls::stream<beat_t> &s2b,
                  hls::stream<beat_t> &s3a, hls::stream<beat_t> &s3b) {
    int idle = 0;
    beat_t v;
drain_loop:
    while (idle < DRAIN_IDLE_CYCLES) {
#pragma HLS PIPELINE II = 1
        bool any = false;
        if (s0a.read_nb(v)) any = true;
        if (s0b.read_nb(v)) any = true;
        if (s1a.read_nb(v)) any = true;
        if (s1b.read_nb(v)) any = true;
        if (s2a.read_nb(v)) any = true;
        if (s2b.read_nb(v)) any = true;
        if (s3a.read_nb(v)) any = true;
        if (s3b.read_nb(v)) any = true;
        idle = any ? 0 : idle + 1;
    }
}

static void load_x(const beat_t *x, int k,
                   xblock_t xq0[MAX_NB], xblock_t xq1[MAX_NB], xblock_t xq2[MAX_NB], xblock_t xq3[MAX_NB],
                   xblock_t xq4[MAX_NB], xblock_t xq5[MAX_NB], xblock_t xq6[MAX_NB], xblock_t xq7[MAX_NB],
                   float xd0[MAX_NB], float xd1[MAX_NB], float xd2[MAX_NB], float xd3[MAX_NB],
                   float xd4[MAX_NB], float xd5[MAX_NB], float xd6[MAX_NB], float xd7[MAX_NB]) {
    const int nb = k / QK;
xq_loop:
    for (int i = 0; i < nb; ++i) {
#pragma HLS PIPELINE II = 2
        xblock_t v;
        v.range(127, 0) = x[2 * i];
        v.range(255, 128) = x[2 * i + 1];
        xq0[i] = v; xq1[i] = v; xq2[i] = v; xq3[i] = v;
        xq4[i] = v; xq5[i] = v; xq6[i] = v; xq7[i] = v;
    }
    const int base = 2 * nb;
xd_loop:
    for (int i = 0; i < nb / 4; ++i) {
#pragma HLS PIPELINE II = 4
        beat_t v = x[base + i];
        for (int j = 0; j < 4; ++j) {
#pragma HLS UNROLL
            ap_uint<32> bits = v.range(32 * j + 31, 32 * j);
            float f = *reinterpret_cast<float *>(&bits);
            xd0[4 * i + j] = f; xd1[4 * i + j] = f; xd2[4 * i + j] = f; xd3[4 * i + j] = f;
            xd4[4 * i + j] = f; xd5[4 * i + j] = f; xd6[4 * i + j] = f; xd7[4 * i + j] = f;
        }
    }
}

extern "C" void gemv_mac(
    hls::stream<beat_t> &s0a, hls::stream<beat_t> &s0b,
    hls::stream<beat_t> &s1a, hls::stream<beat_t> &s1b,
    hls::stream<beat_t> &s2a, hls::stream<beat_t> &s2b,
    hls::stream<beat_t> &s3a, hls::stream<beat_t> &s3b,
    const beat_t *x,
    beat_t *y0a, beat_t *y0b, beat_t *y1a, beat_t *y1b,
    beat_t *y2a, beat_t *y2b, beat_t *y3a, beat_t *y3b,
    int k, int rows_per_mac) {
#pragma HLS INTERFACE axis port = s0a
#pragma HLS INTERFACE axis port = s0b
#pragma HLS INTERFACE axis port = s1a
#pragma HLS INTERFACE axis port = s1b
#pragma HLS INTERFACE axis port = s2a
#pragma HLS INTERFACE axis port = s2b
#pragma HLS INTERFACE axis port = s3a
#pragma HLS INTERFACE axis port = s3b
#pragma HLS INTERFACE m_axi port = x bundle = gmemx offset = slave max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = y0a bundle = gmemy0a offset = slave max_write_burst_length = 16
#pragma HLS INTERFACE m_axi port = y0b bundle = gmemy0b offset = slave max_write_burst_length = 16
#pragma HLS INTERFACE m_axi port = y1a bundle = gmemy1a offset = slave max_write_burst_length = 16
#pragma HLS INTERFACE m_axi port = y1b bundle = gmemy1b offset = slave max_write_burst_length = 16
#pragma HLS INTERFACE m_axi port = y2a bundle = gmemy2a offset = slave max_write_burst_length = 16
#pragma HLS INTERFACE m_axi port = y2b bundle = gmemy2b offset = slave max_write_burst_length = 16
#pragma HLS INTERFACE m_axi port = y3a bundle = gmemy3a offset = slave max_write_burst_length = 16
#pragma HLS INTERFACE m_axi port = y3b bundle = gmemy3b offset = slave max_write_burst_length = 16
#pragma HLS INTERFACE s_axilite port = k
#pragma HLS INTERFACE s_axilite port = rows_per_mac
#pragma HLS INTERFACE s_axilite port = return

    static xblock_t xq0[MAX_NB]; static float xd0[MAX_NB];
#pragma HLS BIND_STORAGE variable = xq0 type = ram_1p impl = bram
    static xblock_t xq1[MAX_NB]; static float xd1[MAX_NB];
#pragma HLS BIND_STORAGE variable = xq1 type = ram_1p impl = bram
    static xblock_t xq2[MAX_NB]; static float xd2[MAX_NB];
#pragma HLS BIND_STORAGE variable = xq2 type = ram_1p impl = bram
    static xblock_t xq3[MAX_NB]; static float xd3[MAX_NB];
#pragma HLS BIND_STORAGE variable = xq3 type = ram_1p impl = bram
    static xblock_t xq4[MAX_NB]; static float xd4[MAX_NB];
#pragma HLS BIND_STORAGE variable = xq4 type = ram_1p impl = bram
    static xblock_t xq5[MAX_NB]; static float xd5[MAX_NB];
#pragma HLS BIND_STORAGE variable = xq5 type = ram_1p impl = bram
    static xblock_t xq6[MAX_NB]; static float xd6[MAX_NB];
#pragma HLS BIND_STORAGE variable = xq6 type = ram_1p impl = bram
    static xblock_t xq7[MAX_NB]; static float xd7[MAX_NB];
#pragma HLS BIND_STORAGE variable = xq7 type = ram_1p impl = bram

    if (rows_per_mac == 0) {
        drain(s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b);
    } else {
        load_x(x, k, xq0, xq1, xq2, xq3, xq4, xq5, xq6, xq7, xd0, xd1, xd2, xd3, xd4, xd5, xd6, xd7);
        engine(s0a, xq0, xd0, k, rows_per_mac, y0a);
        engine(s0b, xq1, xd1, k, rows_per_mac, y0b);
        engine(s1a, xq2, xd2, k, rows_per_mac, y1a);
        engine(s1b, xq3, xd3, k, rows_per_mac, y1b);
        engine(s2a, xq4, xd4, k, rows_per_mac, y2a);
        engine(s2b, xq5, xd5, k, rows_per_mac, y2b);
        engine(s3a, xq6, xd6, k, rows_per_mac, y3a);
        engine(s3b, xq7, xd7, k, rows_per_mac, y3b);
    }
}
