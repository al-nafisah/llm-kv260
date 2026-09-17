#include "gemv_dev.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "xgemv_q4_0_hw.h"

#define AP_START 0x1
#define AP_DONE  0x2
#define AP_IDLE  0x4

static int find_uio(const char *wanted, char *dev, size_t devlen) {
    DIR *d = opendir("/sys/class/uio");
    if (!d) return -1;
    struct dirent *e;
    int found = -1;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "uio", 3)) continue;
        char path[256], name[64] = {0};
        snprintf(path, sizeof path, "/sys/class/uio/%s/name", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        if (fgets(name, sizeof name, f)) name[strcspn(name, "\n")] = 0;
        fclose(f);
        if (!strcmp(name, wanted)) { snprintf(dev, devlen, "/dev/%s", e->d_name); found = 0; break; }
    }
    closedir(d);
    return found;
}

int gemv_dev_open_named(gemv_dev_t *d, const char *name, int quiet) {
    char dev[64];
    if (find_uio(name, dev, sizeof dev)) {
        if (!quiet) fprintf(stderr, "gemv_dev: no UIO device named %s (overlay not loaded?)\n", name);
        return -1;
    }
    d->fd = open(dev, O_RDWR | O_SYNC);
    if (d->fd < 0) { perror(dev); return -1; }
    d->regs = mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, d->fd, 0);
    if (d->regs == MAP_FAILED) { perror("mmap uio"); close(d->fd); return -1; }
    return 0;
}

int gemv_dev_open(gemv_dev_t *d) { return gemv_dev_open_named(d, "gemv_q4_0", 0); }

void gemv_dev_close(gemv_dev_t *d) {
    if (d->regs) munmap((void *)d->regs, 0x10000);
    if (d->fd > 0) close(d->fd);
    d->regs = NULL;
    d->fd = -1;
}

static inline void wr32(gemv_dev_t *d, uint32_t off, uint32_t v) { d->regs[off / 4] = v; }
static inline uint32_t rd32(gemv_dev_t *d, uint32_t off) { return d->regs[off / 4]; }
static inline void wr64(gemv_dev_t *d, uint32_t off, uint64_t v) {
    wr32(d, off, (uint32_t)v);
    wr32(d, off + 4, (uint32_t)(v >> 32));
}

void gemv_dev_start4(gemv_dev_t *d, const uint64_t w[4], uint64_t x, uint64_t y, int k, int rows) {
    const int rpe = rows / 4;
    const uint64_t ystride = (uint64_t)rpe * 4;
    wr64(d, XGEMV_Q4_0_CONTROL_ADDR_W0_DATA, w[0]);
    wr64(d, XGEMV_Q4_0_CONTROL_ADDR_W1_DATA, w[1]);
    wr64(d, XGEMV_Q4_0_CONTROL_ADDR_W2_DATA, w[2]);
    wr64(d, XGEMV_Q4_0_CONTROL_ADDR_W3_DATA, w[3]);
    wr64(d, XGEMV_Q4_0_CONTROL_ADDR_X_DATA, x);
    wr64(d, XGEMV_Q4_0_CONTROL_ADDR_Y0_DATA, y);
    wr64(d, XGEMV_Q4_0_CONTROL_ADDR_Y1_DATA, y + ystride);
    wr64(d, XGEMV_Q4_0_CONTROL_ADDR_Y2_DATA, y + 2 * ystride);
    wr64(d, XGEMV_Q4_0_CONTROL_ADDR_Y3_DATA, y + 3 * ystride);
    wr32(d, XGEMV_Q4_0_CONTROL_ADDR_K_DATA, (uint32_t)k);
    wr32(d, XGEMV_Q4_0_CONTROL_ADDR_ROWS_PER_ENGINE_DATA, (uint32_t)rpe);
    __asm__ volatile("dsb sy" ::: "memory");
    wr32(d, XGEMV_Q4_0_CONTROL_ADDR_AP_CTRL, AP_START);
}

void gemv_dev_start(gemv_dev_t *d, uint64_t w, uint64_t x, uint64_t y, int k, int rows) {
    const uint64_t wstride = (uint64_t)(rows / 4) * k / 32 * 18;   // bytes per engine's rows
    const uint64_t w4[4] = {w, w + wstride, w + 2 * wstride, w + 3 * wstride};
    gemv_dev_start4(d, w4, x, y, k, rows);
}

void gemv_dev_wait(gemv_dev_t *d) {
    // ap_done is sticky until read (COR), so a single poll loop is enough.
    while (!(rd32(d, XGEMV_Q4_0_CONTROL_ADDR_AP_CTRL) & AP_DONE)) { }
}

void gemv_dev_run(gemv_dev_t *d, uint64_t w, uint64_t x, uint64_t y, int k, int rows) {
    gemv_dev_start(d, w, x, y, k, rows);
    gemv_dev_wait(d);
}
