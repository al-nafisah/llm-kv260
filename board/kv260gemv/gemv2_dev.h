// gemv2: the fetch kernel, the eight UltraRAM FIFOs and the MAC kernel
// (hls/gemv2/). Two UIO devices, "gemv_fetch" and "gemv_mac"; register
// offsets from the two HLS-generated headers copied next to this file.
//
// A call is two starts: gemv2_fetch_start(W) as soon as W is known (it
// streams W's bytes into the FIFOs and stalls when they are full), then
// gemv2_mac_start(x, y) when x is ready. The MAC drains whatever the fetch
// put in, so the two must be issued for the same matrix, in order, with
// nothing in between: the fetch of matrix n+1 goes after the MAC of n.
//
// Layout: a [rows, k] matrix is four port regions of rows/4 rows (any
// addresses; stagger them by 16 KB modulo 64 KB, see gemv_dev.h). Inside a
// region, row i pairs with row i + rows/8 and their 144-byte chunks
// alternate: gemv2_chunk_offset gives the byte offset. rows % 32 == 0.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "gemv_dev.h"

typedef struct {
    gemv_dev_t fetch, mac;
} gemv2_dev_t;

#define GEMV2_CHUNK_BYTES 144

int gemv2_dev_open(gemv2_dev_t *d);   // -1 (quietly) when the gemv2 overlay is not loaded
void gemv2_dev_close(gemv2_dev_t *d);

// Port p = r / (rows/4); returns the offset of chunk c of row r inside p's region.
static inline size_t gemv2_chunk_offset(int r, int c, int rows, int k, int *port) {
    const int R = rows / 4, H = R / 2, nch = k / 256;
    const int i = (r % R) % H, which = (r % R) / H;
    *port = r / R;
    return (((size_t)i * nch + c) * 2 + which) * GEMV2_CHUNK_BYTES;
}

void gemv2_fetch_start(gemv2_dev_t *d, const uint64_t w[4], int k, int rows);  // waits for a previous fetch first
void gemv2_fetch_wait(gemv2_dev_t *d);
void gemv2_mac_start(gemv2_dev_t *d, uint64_t x, uint64_t y, int k, int rows);
void gemv2_mac_wait(gemv2_dev_t *d);
int gemv2_mac_idle(gemv2_dev_t *d);
// Empty FIFOs that still hold a previous process's prefetch. Blocks ~16 us plus whatever a stalled fetch still has to deliver.
void gemv2_drain(gemv2_dev_t *d);
