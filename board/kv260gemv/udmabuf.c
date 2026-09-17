#include "udmabuf.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static int read_sysfs_u64(const char *name, const char *attr, uint64_t *out) {
    char path[256];
    snprintf(path, sizeof path, "/sys/class/u-dma-buf/%s/%s", name, attr);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int ok = fscanf(f, "%lli", (long long *)out) == 1;
    fclose(f);
    return ok ? 0 : -1;
}

int udmabuf_open(udmabuf_t *b, const char *name) {
    memset(b, 0, sizeof *b);
    uint64_t size;
    if (read_sysfs_u64(name, "phys_addr", &b->phys) || read_sysfs_u64(name, "size", &size)) {
        fprintf(stderr, "udmabuf: no /sys/class/u-dma-buf/%s (is u-dma-buf.ko loaded?)\n", name);
        return -1;
    }
    char dev[64];
    snprintf(dev, sizeof dev, "/dev/%s", name);
    b->fd = open(dev, O_RDWR);   // no O_SYNC: we want the cached mapping
    if (b->fd < 0) { perror(dev); return -1; }
    b->base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, b->fd, 0);
    if (b->base == MAP_FAILED) { perror("mmap udmabuf"); close(b->fd); return -1; }
    b->size = size;
    return 0;
}

void udmabuf_close(udmabuf_t *b) {
    if (b->base) munmap(b->base, b->size);
    if (b->fd > 0) close(b->fd);
    memset(b, 0, sizeof *b);
}

void *udmabuf_alloc(udmabuf_t *b, size_t size, size_t align) {
    size_t start = (b->used + align - 1) & ~(align - 1);
    if (start + size > b->size) return NULL;
    b->used = start + size;
    return b->base + start;
}

// Cortex-A53: 64-byte lines. "dc cvac" cleans, "dc civac" cleans and
// invalidates. A plain invalidate ("dc ivac") is EL1-only, but the engine's
// output buffers are never dirty in the CPU cache, so clean+invalidate is
// equivalent for them and it is what EL0 is allowed.
#define LINE 64

void udmabuf_clean(const void *p, size_t len) {
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)(LINE - 1);
    uintptr_t end = (uintptr_t)p + len;
    for (; a < end; a += LINE) __asm__ volatile("dc cvac, %0" ::"r"(a) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
}

void udmabuf_invalidate(const void *p, size_t len) {
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)(LINE - 1);
    uintptr_t end = (uintptr_t)p + len;
    for (; a < end; a += LINE) __asm__ volatile("dc civac, %0" ::"r"(a) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
}
