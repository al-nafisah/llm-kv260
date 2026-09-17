// gemv_fetch: see gemv2.h. Pure DMA: one burst reader per HP port, chunks
// dealt alternately to the port's two FIFOs. Started as soon as the next
// matrix is known; blocks on the FIFOs until the MAC kernel drains them.
#include "gemv2.h"

static void reader(const beat_t *w, int beats, hls::stream<beat_t> &a, hls::stream<beat_t> &b) {
    ap_uint<4> i = 0;   // beat within the chunk
    bool sel = false;   // which FIFO gets this chunk
reader_loop:
    for (int t = 0; t < beats; ++t) {
#pragma HLS PIPELINE II = 1
        beat_t v = w[t];
        if (sel) b.write(v); else a.write(v);
        if (i == BEATS_PER_CHUNK - 1) { i = 0; sel = !sel; } else { ++i; }
    }
}

extern "C" void gemv_fetch(
    const beat_t *w0, const beat_t *w1, const beat_t *w2, const beat_t *w3,
    int beats_per_port,
    hls::stream<beat_t> &s0a, hls::stream<beat_t> &s0b,
    hls::stream<beat_t> &s1a, hls::stream<beat_t> &s1b,
    hls::stream<beat_t> &s2a, hls::stream<beat_t> &s2b,
    hls::stream<beat_t> &s3a, hls::stream<beat_t> &s3b) {
    // Read FIFO per port: 16 outstanding x 64 beats x 16 B, 8x the
    // bandwidth-delay product and small enough for the block RAM.
#pragma HLS INTERFACE m_axi port = w0 bundle = gmem0 offset = slave max_read_burst_length = 64 num_read_outstanding = 16 latency = 64
#pragma HLS INTERFACE m_axi port = w1 bundle = gmem1 offset = slave max_read_burst_length = 64 num_read_outstanding = 16 latency = 64
#pragma HLS INTERFACE m_axi port = w2 bundle = gmem2 offset = slave max_read_burst_length = 64 num_read_outstanding = 16 latency = 64
#pragma HLS INTERFACE m_axi port = w3 bundle = gmem3 offset = slave max_read_burst_length = 64 num_read_outstanding = 16 latency = 64
#pragma HLS INTERFACE axis port = s0a
#pragma HLS INTERFACE axis port = s0b
#pragma HLS INTERFACE axis port = s1a
#pragma HLS INTERFACE axis port = s1b
#pragma HLS INTERFACE axis port = s2a
#pragma HLS INTERFACE axis port = s2b
#pragma HLS INTERFACE axis port = s3a
#pragma HLS INTERFACE axis port = s3b
#pragma HLS INTERFACE s_axilite port = beats_per_port
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW
    reader(w0, beats_per_port, s0a, s0b);
    reader(w1, beats_per_port, s1a, s1b);
    reader(w2, beats_per_port, s2a, s2b);
    reader(w3, beats_per_port, s3a, s3b);
}
