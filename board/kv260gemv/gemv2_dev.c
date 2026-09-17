#include "gemv2_dev.h"

#include <stdio.h>

#include "xgemv_fetch_hw.h"
#include "xgemv_mac_hw.h"

#define AP_START 0x1
#define AP_DONE  0x2
#define AP_IDLE  0x4

static inline void wr32(gemv_dev_t *d, uint32_t off, uint32_t v) { d->regs[off / 4] = v; }
static inline uint32_t rd32(gemv_dev_t *d, uint32_t off) { return d->regs[off / 4]; }
static inline void wr64(gemv_dev_t *d, uint32_t off, uint64_t v) {
    wr32(d, off, (uint32_t)v);
    wr32(d, off + 4, (uint32_t)(v >> 32));
}

int gemv2_dev_open(gemv2_dev_t *d) {
    if (gemv_dev_open_named(&d->fetch, "gemv_fetch", 1)) return -1;
    if (gemv_dev_open_named(&d->mac, "gemv_mac", 0)) { gemv_dev_close(&d->fetch); return -1; }
    return 0;
}

void gemv2_dev_close(gemv2_dev_t *d) {
    gemv_dev_close(&d->fetch);
    gemv_dev_close(&d->mac);
}

void gemv2_fetch_wait(gemv2_dev_t *d) {
    while (!(rd32(&d->fetch, XGEMV_FETCH_CONTROL_ADDR_AP_CTRL) & AP_IDLE)) { }
}

void gemv2_fetch_start(gemv2_dev_t *d, const uint64_t w[4], int k, int rows) {
    gemv2_fetch_wait(d);
    const int beats = rows / 4 * (k * 9 / 256);
    wr64(&d->fetch, XGEMV_FETCH_CONTROL_ADDR_W0_DATA, w[0]);
    wr64(&d->fetch, XGEMV_FETCH_CONTROL_ADDR_W1_DATA, w[1]);
    wr64(&d->fetch, XGEMV_FETCH_CONTROL_ADDR_W2_DATA, w[2]);
    wr64(&d->fetch, XGEMV_FETCH_CONTROL_ADDR_W3_DATA, w[3]);
    wr32(&d->fetch, XGEMV_FETCH_CONTROL_ADDR_BEATS_PER_PORT_DATA, (uint32_t)beats);
    __asm__ volatile("dsb sy" ::: "memory");
    wr32(&d->fetch, XGEMV_FETCH_CONTROL_ADDR_AP_CTRL, AP_START);
}

void gemv2_mac_start(gemv2_dev_t *d, uint64_t x, uint64_t y, int k, int rows) {
    const int rpm = rows / 8;
    const uint64_t ys = (uint64_t)rpm * 4;
    wr64(&d->mac, XGEMV_MAC_CONTROL_ADDR_X_DATA, x);
    wr64(&d->mac, XGEMV_MAC_CONTROL_ADDR_Y0A_DATA, y);
    wr64(&d->mac, XGEMV_MAC_CONTROL_ADDR_Y0B_DATA, y + ys);
    wr64(&d->mac, XGEMV_MAC_CONTROL_ADDR_Y1A_DATA, y + 2 * ys);
    wr64(&d->mac, XGEMV_MAC_CONTROL_ADDR_Y1B_DATA, y + 3 * ys);
    wr64(&d->mac, XGEMV_MAC_CONTROL_ADDR_Y2A_DATA, y + 4 * ys);
    wr64(&d->mac, XGEMV_MAC_CONTROL_ADDR_Y2B_DATA, y + 5 * ys);
    wr64(&d->mac, XGEMV_MAC_CONTROL_ADDR_Y3A_DATA, y + 6 * ys);
    wr64(&d->mac, XGEMV_MAC_CONTROL_ADDR_Y3B_DATA, y + 7 * ys);
    wr32(&d->mac, XGEMV_MAC_CONTROL_ADDR_K_DATA, (uint32_t)k);
    wr32(&d->mac, XGEMV_MAC_CONTROL_ADDR_ROWS_PER_MAC_DATA, (uint32_t)rpm);
    __asm__ volatile("dsb sy" ::: "memory");
    wr32(&d->mac, XGEMV_MAC_CONTROL_ADDR_AP_CTRL, AP_START);
}

void gemv2_mac_wait(gemv2_dev_t *d) {
    while (!(rd32(&d->mac, XGEMV_MAC_CONTROL_ADDR_AP_CTRL) & AP_DONE)) { }
}

int gemv2_mac_idle(gemv2_dev_t *d) { return (rd32(&d->mac, XGEMV_MAC_CONTROL_ADDR_AP_CTRL) & AP_IDLE) != 0; }

void gemv2_drain(gemv2_dev_t *d) {
    wr32(&d->mac, XGEMV_MAC_CONTROL_ADDR_K_DATA, 256);
    wr32(&d->mac, XGEMV_MAC_CONTROL_ADDR_ROWS_PER_MAC_DATA, 0);
    __asm__ volatile("dsb sy" ::: "memory");
    wr32(&d->mac, XGEMV_MAC_CONTROL_ADDR_AP_CTRL, AP_START);
    gemv2_mac_wait(d);
}
