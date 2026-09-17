// The engine's control interface: one UIO device, one register block.
//
// Register offsets come from the header Vitis HLS generates with the IP
// (xgemv_q4_0_hw.h, copied next to this file), so a change to the kernel's
// argument list cannot silently desynchronise the driver.
#pragma once

#include <stdint.h>

typedef struct {
    int fd;
    volatile uint32_t *regs;
} gemv_dev_t;

int gemv_dev_open(gemv_dev_t *d);   // finds the UIO device named "gemv_q4_0"
int gemv_dev_open_named(gemv_dev_t *d, const char *name, int quiet);   // any HLS kernel's UIO block
void gemv_dev_close(gemv_dev_t *d);

// y[rows] = W[rows, k] x. Physical addresses. `w` is the first byte of the
// Q4_0 rows exactly as ggml lays them out; `x` is int8[k] then float32[k/32]
// (pack_x_for_engine); `y` receives rows float32. Blocks until done.
// Requires k % 256 == 0, k <= 8192, rows % 16 == 0.
void gemv_dev_run(gemv_dev_t *d, uint64_t w, uint64_t x, uint64_t y, int k, int rows);

// Split form, for a caller that overlaps its own work with the engine.
void gemv_dev_start(gemv_dev_t *d, uint64_t w, uint64_t x, uint64_t y, int k, int rows);
void gemv_dev_wait(gemv_dev_t *d);

// Each engine's quarter of the rows at its own address. The four streams
// share the DDR banks, and quarters that start at the same offset modulo
// 64 KB collide there: 10.3 GB/s against 12.7 when the starts are
// staggered by 16 KB (buildlog 0012). gemv_dev_start places quarter m at
// w + m * quarter_bytes, which for every Llama shape is such a collision;
// a caller that lays the matrix out itself passes the four starts here.
void gemv_dev_start4(gemv_dev_t *d, const uint64_t w[4], uint64_t x, uint64_t y, int k, int rows);
