// Physically contiguous, CPU-cached memory the engine can stream from.
//
// u-dma-buf (ikwzm) allocates it from CMA at module load and exposes the
// physical address in sysfs. The mapping is cached, so the CPU is fast, and
// coherence is the caller's job: `udmabuf_clean` before the engine reads
// what the CPU wrote, `udmabuf_invalidate` before the CPU reads what the
// engine wrote. Both are userspace cache-maintenance instructions, which
// arm64 Linux permits from EL0, so they cost nanoseconds per line and no
// syscall.
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int fd;
    uint8_t *base;      // CPU virtual address
    uint64_t phys;      // what the engine's AXI masters are given
    size_t size;
    size_t used;        // bump allocator watermark
} udmabuf_t;

int udmabuf_open(udmabuf_t *b, const char *name);   // e.g. "udmabuf0"
void udmabuf_close(udmabuf_t *b);

// Bump-allocate `size` bytes, aligned to `align` (a power of two).
// Returns NULL when full. There is no free: a model is loaded once.
void *udmabuf_alloc(udmabuf_t *b, size_t size, size_t align);

static inline uint64_t udmabuf_phys(const udmabuf_t *b, const void *p) {
    return b->phys + (uint64_t)((const uint8_t *)p - b->base);
}

// Cache maintenance over [p, p+len).
void udmabuf_clean(const void *p, size_t len);        // CPU wrote, device will read
void udmabuf_invalidate(const void *p, size_t len);   // device wrote, CPU will read
