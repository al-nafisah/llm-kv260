// gemv2: the phase 5 engine. Two kernels and eight UltraRAM FIFOs between
// them, so that the weight stream for the next matrix starts while the CPU
// is still busy with the exchange or attention (docs/buildlog/0012).
//
//   gemv_fetch  four DRAM readers (HP0..3). Port p's rows are laid out as
//               row pairs with their 144-byte chunks interleaved, and the
//               reader sends alternate chunks to FIFO pa and FIFO pb.
//   gemv_mac    eight MAC units (two per port), each running the datapath in
//               int4_core/gemv_core.h over one FIFO and writing its rows of y.
//
// Two MAC units per port because a FIFO full of prefetched bytes only helps
// if it drains faster than DRAM refills it: one unit per port consumes
// 16 GB/s against a 12.7 GB/s DRAM, two consume 32.
//
// rows_per_mac == 0 is the drain command: one loop reads all eight FIFOs
// until every one has been empty for 4096 cycles. A process that starts on
// a board whose FIFOs still hold the previous process's prefetch runs it
// once (it must read all eight at once: see drain() in gemv_mac.cpp).
#pragma once

#include "../int4_core/q4_0.h"

#include <hls_stream.h>

static const int PORTS = 4;
static const int MACS_PER_PORT = 2;
static const int MACS = PORTS * MACS_PER_PORT;
static const int DRAIN_IDLE_CYCLES = 4096;
static const int CHUNK_BYTES = BEATS_PER_CHUNK * 16;   // 144

extern "C" void gemv_fetch(
    const beat_t *w0, const beat_t *w1, const beat_t *w2, const beat_t *w3,
    int beats_per_port,   // rows_per_port * k * 9 / 256
    hls::stream<beat_t> &s0a, hls::stream<beat_t> &s0b,
    hls::stream<beat_t> &s1a, hls::stream<beat_t> &s1b,
    hls::stream<beat_t> &s2a, hls::stream<beat_t> &s2b,
    hls::stream<beat_t> &s3a, hls::stream<beat_t> &s3b);

extern "C" void gemv_mac(
    hls::stream<beat_t> &s0a, hls::stream<beat_t> &s0b,
    hls::stream<beat_t> &s1a, hls::stream<beat_t> &s1b,
    hls::stream<beat_t> &s2a, hls::stream<beat_t> &s2b,
    hls::stream<beat_t> &s3a, hls::stream<beat_t> &s3b,
    const beat_t *x,
    beat_t *y0a, beat_t *y0b, beat_t *y1a, beat_t *y1b,
    beat_t *y2a, beat_t *y2b, beat_t *y3a, beat_t *y3b,
    int k,              // multiple of 256, <= MAX_K
    int rows_per_mac);  // multiple of 4; 0 = drain
