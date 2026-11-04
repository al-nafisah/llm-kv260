// tp_llama: one Llama decode stream, tensor-parallel across KV260 boards.
//
// Every rank holds a 1/n shard of every weight matrix in its contiguous
// memory and runs it on its own fabric engine. Column-parallel matrices
// (q, k, v, gate, up, lm_head) shard by output rows, which in ggml's Q4_0
// row-major layout is a contiguous byte range. Row-parallel matrices
// (attn_output, ffn_down) shard by input columns, so each rank's partial
// product has to be summed with the others': one all-reduce after
// attention and one after the MLP, 32 per token for this model, plus an
// argmax exchange for the next token. Attention itself needs no traffic:
// rank r owns query heads 8r..8r+7 and the two KV heads they attend to.
//
// The all-reduce sums the ranks' partials in rank order on every rank, so
// the replicated hidden state stays bit-identical everywhere. Everything
// between matmuls runs on one A53 core in NEON intrinsics -- no scheduler,
// no thread pool starts, no splits -- which is where the single-board
// llama.cpp path lost 40% of its token. GCC 11 at -O2 vectorizes none of
// these loops on its own; written scalar, attention alone cost 0.12 ms per
// context position per token (47 ms at position 400).
//
//   tp_llama --model m.gguf --rank 0 --nranks 4 --peers ip0,ip1,ip2,ip3 \
//            --prompt "..." --n 64
//
// Weights: ggml Q4_0 (all tensors; the lm_head is the tied token_embd).
// Tokenizer: llama.cpp's own, through libllama (vocab only).

#include <arm_neon.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/mman.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "httplib.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include "ggml.h"
#include "gguf.h"
#include "ggml-quants.h"
#include "llama.h"

extern "C" {
#include "gemv_dev.h"
#include "gemv2_dev.h"
#include "udmabuf.h"
}

#define DIE(...) do { fprintf(stderr, "tp_llama: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

static double now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ------------------------------------------------------------------ memory

// The three u-dma-buf pools, largest first, as one allocator.
struct pools_t {
    std::vector<udmabuf_t> p;
    void open() {
        for (const char * name : {"udmabuf0", "udmabuf1", "udmabuf2"}) {
            udmabuf_t b;
            if (udmabuf_open(&b, name) == 0) p.push_back(b);
        }
        if (p.empty()) DIE("no u-dma-buf pools (load-overlay.sh?)");
    }
    void * alloc(size_t size, uint64_t * phys, size_t align = 4096) {
        for (auto & b : p) {
            void * v = udmabuf_alloc(&b, size, align);
            if (v) { *phys = udmabuf_phys(&b, v); return v; }
        }
        DIE("out of contiguous memory allocating %zu MB", size >> 20);
    }
};

// A Q4_0 matrix shard in engine memory: four quarters of its rows, one per
// engine, each in its own 64 KB-aligned allocation, quarters 1 and 3 used from
// 16 KB in. The four streams then do not all start at the same offset modulo
// 64 KB: 12.7 GB/s instead of 10.3 when they all do (buildlog 0012).
//
// Two layouts inside a quarter: gemv_q4_0 streams rows in order; gemv2
// pairs row i with row i + rows/8 and alternates their 144-byte chunks
// (gemv2_dev.h), so everything is copied chunk by chunk through chunk().
struct qmat {
    uint8_t * q[4] = {};
    uint64_t qphys[4] = {};
    int rows = 0, k = 0;
    bool v2 = false;
    size_t bytes() const { return (size_t) rows * k / 32 * 18; }
    size_t row_bytes() const { return (size_t) k / 32 * 18; }
    uint8_t * chunk(int r, int c) const {
        const int R = rows / 4;
        if (!v2) return q[r / R] + (size_t) (r % R) * row_bytes() + (size_t) c * GEMV2_CHUNK_BYTES;
        int p;
        const size_t o = gemv2_chunk_offset(r, c, rows, k, &p);
        return q[p] + o;
    }
};

// ------------------------------------------------------------------- model

struct hparams {
    int n_layer, n_embd, n_ff, n_head, n_head_kv, n_vocab, head_dim;
    float rope_base, rms_eps;
};

struct layer_w {
    qmat qkv;      // rows: q shard, then k shard, then v shard (one engine call)
    qmat o;        // [n_embd, n_embd/n]        row-parallel
    qmat gateup;   // rows: gate shard, then up shard
    qmat down;     // [n_embd, n_ff/n]          row-parallel
    const float * attn_norm;
    const float * ffn_norm;
};

struct model_t {
    hparams hp;
    int rank, nranks;
    std::vector<layer_w> layers;
    qmat lm_head;                 // [n_vocab/n, n_embd]
    const float * output_norm;
    const float * rope_freqs;     // 32 factors, or null
    const uint8_t * token_embd;   // Q4_0, in the mmapped file
    // per-rank shard sizes
    int q_rows, kv_rows, ff_rows, vocab_rows, o_k, down_k;
};

struct gguf_file {
    gguf_context * gg = nullptr;
    ggml_context * meta = nullptr;
    uint8_t * map = nullptr;
    size_t size = 0;
    void open(const char * path) {
        ggml_init_params ip = { /*mem_size*/ 0, nullptr, true };
        gguf_init_params gp = { /*no_alloc*/ true, &meta };
        ip.mem_size = 1024 * 1024;
        gp.ctx = &meta;
        gg = gguf_init_from_file(path, gp);
        if (!gg) DIE("cannot read %s", path);
        int fd = ::open(path, O_RDONLY);
        struct stat st;
        fstat(fd, &st);
        size = st.st_size;
        map = (uint8_t *) mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (map == MAP_FAILED) DIE("mmap %s", path);
    }
    const ggml_tensor * tensor(const char * name) const {
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        if (!t) DIE("tensor %s missing", name);
        return t;
    }
    const uint8_t * data(const char * name) const {
        int64_t i = gguf_find_tensor(gg, name);
        if (i < 0) DIE("tensor %s missing", name);
        return map + gguf_get_data_offset(gg) + gguf_get_tensor_offset(gg, i);
    }
    uint32_t u32(const char * key) const {
        int64_t i = gguf_find_key(gg, key);
        if (i < 0) DIE("key %s missing", key);
        return gguf_get_val_u32(gg, i);
    }
    float f32(const char * key, float dflt) const {
        int64_t i = gguf_find_key(gg, key);
        return i < 0 ? dflt : gguf_get_val_f32(gg, i);
    }
};

// Copy rows [r0, r0+rows) of a Q4_0 [N, K] matrix to rows [dst_row, ...) of m.
static void copy_rows(qmat & m, const uint8_t * src, int k, int r0, int rows, int dst_row) {
    const size_t rb = (size_t) k / 32 * 18;
    const int nch = k / 256;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < nch; ++c)
            memcpy(m.chunk(dst_row + r, c), src + (size_t) (r0 + r) * rb + (size_t) c * GEMV2_CHUNK_BYTES, GEMV2_CHUNK_BYTES);
}

// Copy columns [c0, c0+kk) of every row of a Q4_0 [N, K] matrix: 18-byte
// blocks sliced out of each row. c0 and kk are multiples of 256.
static void copy_cols(qmat & m, const uint8_t * src, int k, int n, int c0, int kk) {
    const size_t rb = (size_t) k / 32 * 18, off = (size_t) c0 / 32 * 18;
    const int nch = kk / 256;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < nch; ++c)
            memcpy(m.chunk(r, c), src + (size_t) r * rb + off + (size_t) c * GEMV2_CHUNK_BYTES, GEMV2_CHUNK_BYTES);
}

static qmat new_mat(pools_t & pools, int rows, int k, bool v2) {
    qmat m;
    m.rows = rows;
    m.k = k;
    m.v2 = v2;
    const size_t quarter = m.bytes() / 4, stagger = 16 << 10;
    for (int e = 0; e < 4; ++e) {
        // 64 KB-aligned allocation, quarter e used from offset 0 or 16 KB.
        const size_t off = (e & 1) * stagger;
        uint64_t phys;
        uint8_t * base = (uint8_t *) pools.alloc(quarter + off, &phys, 64 << 10);
        m.q[e] = base + off;
        m.qphys[e] = phys + off;
    }
    return m;
}

static void load_model(model_t & M, gguf_file & f, pools_t & pools, bool v2) {
    hparams & h = M.hp;
    h.n_layer = f.u32("llama.block_count");
    h.n_embd = f.u32("llama.embedding_length");
    h.n_ff = f.u32("llama.feed_forward_length");
    h.n_head = f.u32("llama.attention.head_count");
    h.n_head_kv = f.u32("llama.attention.head_count_kv");
    h.n_vocab = f.u32("llama.vocab_size");
    h.head_dim = h.n_embd / h.n_head;
    h.rope_base = f.f32("llama.rope.freq_base", 10000.0f);
    h.rms_eps = f.f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);

    const int n = M.nranks;
    if (h.n_head % n || h.n_head_kv % n || h.n_ff % n || h.n_vocab % n) DIE("%d ranks do not divide this model", n);
    M.q_rows = h.n_embd / n;
    M.kv_rows = h.n_head_kv * h.head_dim / n;
    M.ff_rows = h.n_ff / n;
    M.vocab_rows = h.n_vocab / n;
    M.o_k = h.n_embd / n;
    M.down_k = h.n_ff / n;
    for (int v : {M.q_rows + 2 * M.kv_rows, h.n_embd, 2 * M.ff_rows, M.vocab_rows}) if (v % 32) DIE("shard rows %d not a multiple of 32", v);
    for (int v : {h.n_embd, M.o_k, M.down_k}) if (v % 256 || v > 8192) DIE("shard width %d unusable by the engine", v);

    const int r = M.rank;
    M.layers.resize(h.n_layer);
    size_t total = 0;
    for (int l = 0; l < h.n_layer; ++l) {
        layer_w & L = M.layers[l];
        char name[64];
        auto T = [&](const char * what) { snprintf(name, sizeof name, "blk.%d.%s.weight", l, what); return f.data(name); };
        auto F = [&](const char * what) { snprintf(name, sizeof name, "blk.%d.%s.weight", l, what); return (const float *) f.data(name); };

        L.qkv = new_mat(pools, M.q_rows + 2 * M.kv_rows, h.n_embd, v2);
        copy_rows(L.qkv, T("attn_q"), h.n_embd, r * M.q_rows, M.q_rows, 0);
        copy_rows(L.qkv, T("attn_k"), h.n_embd, r * M.kv_rows, M.kv_rows, M.q_rows);
        copy_rows(L.qkv, T("attn_v"), h.n_embd, r * M.kv_rows, M.kv_rows, M.q_rows + M.kv_rows);

        L.o = new_mat(pools, h.n_embd, M.o_k, v2);
        copy_cols(L.o, T("attn_output"), h.n_embd, h.n_embd, r * M.o_k, M.o_k);

        L.gateup = new_mat(pools, 2 * M.ff_rows, h.n_embd, v2);
        copy_rows(L.gateup, T("ffn_gate"), h.n_embd, r * M.ff_rows, M.ff_rows, 0);
        copy_rows(L.gateup, T("ffn_up"), h.n_embd, r * M.ff_rows, M.ff_rows, M.ff_rows);

        L.down = new_mat(pools, h.n_embd, M.down_k, v2);
        copy_cols(L.down, T("ffn_down"), h.n_ff, h.n_embd, r * M.down_k, M.down_k);

        L.attn_norm = F("attn_norm");
        L.ffn_norm = F("ffn_norm");
        total += L.qkv.bytes() + L.o.bytes() + L.gateup.bytes() + L.down.bytes();
    }
    M.token_embd = f.data("token_embd.weight");
    M.lm_head = new_mat(pools, M.vocab_rows, h.n_embd, v2);
    copy_rows(M.lm_head, M.token_embd, h.n_embd, r * M.vocab_rows, M.vocab_rows, 0);
    total += M.lm_head.bytes();
    M.output_norm = (const float *) f.data("output_norm.weight");
    M.rope_freqs = gguf_find_tensor(f.gg, "rope_freqs.weight") >= 0 ? (const float *) f.data("rope_freqs.weight") : nullptr;

    // Everything the engine reads was written by the CPU: clean once.
    for (auto & p : pools.p) udmabuf_clean(p.base, p.used);
    fprintf(stderr, "rank %d: %d layers, shard %.1f MB of weights in engine memory\n", r, h.n_layer, total / 1e6);
}

// ------------------------------------------------------------------ engine

// Either engine: gemv2 (fetch kernel, FIFOs, MAC kernel) when its overlay
// is loaded, else gemv_q4_0. With gemv2 the caller names the matrix that
// comes next, and its bytes start streaming into the FIFOs the moment this
// call's MAC is done, while the CPU does attention or the exchange.
struct engine_t {
    bool v2 = false;
    gemv_dev_t dev;
    gemv2_dev_t dev2;
    const qmat * pending = nullptr;    // gemv2: the matrix whose bytes are in (or on their way into) the FIFOs
    uint8_t * x; uint64_t x_phys;      // int8[K] + float[K/32]
    float * y; uint64_t y_phys;        // up to vocab_rows floats
    size_t y_cap;
    void probe() {
        if (gemv2_dev_open(&dev2) == 0) {
            v2 = true;
            if (!gemv2_mac_idle(&dev2)) DIE("gemv2 MAC kernel is busy: reload the overlay");
            gemv2_drain(&dev2);   // whatever the previous process left in the FIFOs
        } else if (gemv_dev_open(&dev)) {
            DIE("engine not available");
        }
        fprintf(stderr, "engine: %s\n", v2 ? "gemv2 (fetch + FIFOs + 8 MAC units)" : "gemv_q4_0 (4 engines)");
    }
    void open(pools_t & pools, size_t y_floats) {
        x = (uint8_t *) pools.alloc(8192 + 256 * 4, &x_phys);
        y_cap = y_floats;
        y = (float *) pools.alloc(y_floats * 4, &y_phys);
    }
    void prefetch(const qmat & W) {
        if (!v2 || pending == &W) return;
        if (pending) DIE("engine: a prefetch of another matrix is pending");
        gemv2_fetch_start(&dev2, W.qphys, W.k, W.rows);
        pending = &W;
    }
    // Q8_0 quantization of x into the engine's layout (int8[k], then the
    // float32 scale of each block), the same values as ggml's
    // quantize_row_q8_0_ref: amax/127, the scale rounded through float16,
    // roundf on the quotients (vcvta rounds ties away from zero as roundf
    // does; ggml's own NEON path uses vcvtn, ties to even, and would differ
    // on exact halves). The scalar reference was 30 us of a 77 us call.
    static void quantize_x(const float * xin, int8_t * qs, float * xd, int k) {
        for (int b = 0; b < k / 32; ++b) {
            const float * xb = xin + 32 * b;
            float32x4_t m = vdupq_n_f32(0.0f);
#pragma GCC unroll 8
            for (int j = 0; j < 32; j += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(xb + j)));
            const float amax = vmaxvq_f32(m);
            const float d = amax / 127.0f;
            const float id = d ? 1.0f / d : 0.0f;
            xd[b] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
            const float32x4_t vid = vdupq_n_f32(id);
#pragma GCC unroll 2
            for (int j = 0; j < 32; j += 16) {
                int32x4_t i0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + j), vid));
                int32x4_t i1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + j + 4), vid));
                int32x4_t i2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + j + 8), vid));
                int32x4_t i3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + j + 12), vid));
                int16x8_t s01 = vcombine_s16(vmovn_s32(i0), vmovn_s32(i1));
                int16x8_t s23 = vcombine_s16(vmovn_s32(i2), vmovn_s32(i3));
                vst1q_s8(qs + 32 * b + j, vcombine_s8(vmovn_s16(s01), vmovn_s16(s23)));
            }
        }
    }
    // y = W x. x is float32[k]; result left in this->y (rows of W floats).
    // `next` is the matrix the caller will pass next (gemv2 prefetches it).
    void gemv(const qmat & W, const float * xin, const qmat * next) {
        const int k = W.k, nb = k / 32;
        quantize_x(xin, (int8_t *) x, (float *) (x + k), k);
        udmabuf_clean(x, k + 4 * nb);
        // Before the run as well as after: rope() writes q in place in y,
        // and a dirty line evicted during the next run would land on top of
        // the engine's output (it did: garbage from the first token).
        udmabuf_invalidate(y, (size_t) W.rows * 4);
        if (v2) {
            prefetch(W);            // no-op when it is already pending
            pending = nullptr;
            gemv2_mac_start(&dev2, x_phys, y_phys, k, W.rows);
            gemv2_mac_wait(&dev2);
            if (next) prefetch(*next);
        } else {
            gemv_dev_start4(&dev, W.qphys, x_phys, y_phys, k, W.rows);
            gemv_dev_wait(&dev);
        }
        udmabuf_invalidate(y, (size_t) W.rows * 4);
    }
};

// -------------------------------------------------------------------- comm

// A full mesh of TCP connections. Every round is: write this rank's message
// to every peer, then collect every peer's message with non-blocking reads
// polled round-robin (a blocking read sleeps and is woken once per peer;
// polling keeps the core and cut the 64-byte round from 82 to 69 us). The
// kernel owns reliability and ordering; TCP_NODELAY keeps a message from
// waiting on Nagle. A UDP version with its own resend logic came first and
// lost packets by the hundred under the 18-packet bursts of a four-way
// exchange on these NICs; this one has no loss path at all.
//
// The link is the cost: at 1 Gbit/s a rank's 24 KB out and 24 KB in per
// 8 KB all-gather is ~0.2 ms of wire in a 0.4 ms round, so the all-reduce
// is a reduce-scatter (each rank sums its own quarter, in rank order) and
// an all-gather of the quarters: half the bytes, two rounds, the same
// float32 sums bit for bit (buildlog/0011).
//
// Connection setup: rank r accepts from every higher rank and connects to
// every lower one, so each pair has exactly one socket.
struct comm_t {
    int rank, n;
    std::vector<int> fd;   // fd[r] is the socket to rank r (-1 for self)
    std::vector<int> got;  // bytes received from each peer in the current round

    static void set_opts(int s) {
        int one = 1;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        int buf = 1 << 20;
        setsockopt(s, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
        setsockopt(s, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    }

    void open(int rank_, const std::vector<std::string> & ips, int port) {
        rank = rank_; n = (int) ips.size();
        fd.assign(n, -1); got.assign(n, 0);
        if (n == 1) return;
        int ls = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in me = {};
        me.sin_family = AF_INET; me.sin_port = htons(port); me.sin_addr.s_addr = INADDR_ANY;
        if (bind(ls, (sockaddr *) &me, sizeof me) || listen(ls, n)) DIE("listen on port %d", port);
        // Connect to lower ranks (retrying while they come up).
        for (int r = 0; r < rank; ++r) {
            sockaddr_in a = {};
            a.sin_family = AF_INET; a.sin_port = htons(port);
            if (inet_pton(AF_INET, ips[r].c_str(), &a.sin_addr) != 1) DIE("bad ip %s", ips[r].c_str());
            for (int attempt = 0;; ++attempt) {
                int s = socket(AF_INET, SOCK_STREAM, 0);
                if (connect(s, (sockaddr *) &a, sizeof a) == 0) {
                    set_opts(s);
                    uint8_t me_id = (uint8_t) rank;
                    if (write(s, &me_id, 1) != 1) DIE("handshake to rank %d", r);
                    fd[r] = s;
                    break;
                }
                close(s);
                if (attempt > 600) DIE("rank %d never answered on %s:%d", r, ips[r].c_str(), port);
                usleep(100000);
            }
        }
        // Accept from higher ranks; each announces its rank first.
        for (int r = rank + 1; r < n; ++r) {
            int s = accept(ls, nullptr, nullptr);
            if (s < 0) DIE("accept");
            set_opts(s);
            uint8_t who;
            if (read(s, &who, 1) != 1 || who <= rank || who >= n || fd[who] != -1) DIE("bad handshake");
            fd[who] = s;
        }
        close(ls);
    }

    static void write_all(int s, const uint8_t * p, int len) {
        while (len > 0) {
            ssize_t w = write(s, p, len);
            if (w <= 0) DIE("peer connection lost (write)");
            p += w; len -= (int) w;
        }
    }
    static void read_all(int s, uint8_t * p, int len) {
        while (len > 0) {
            ssize_t r = read(s, p, len);
            if (r <= 0) DIE("peer connection lost (read)");
            p += r; len -= (int) r;
        }
    }

    // One round: send to[r] (len[r] bytes) to every peer r, receive len[r]
    // bytes from it into from[r]. The 1 MB socket buffers take a whole
    // message, so the writes never block on a peer that is still computing.
    void round(const uint8_t * const * to, uint8_t * const * from, const int * len) {
        int remaining = 0;
        for (int r = 0; r < n; ++r) if (r != rank) { write_all(fd[r], to[r], len[r]); got[r] = 0; remaining += len[r] > 0; }
        while (remaining > 0) {
            for (int r = 0; r < n; ++r) {
                if (r == rank || got[r] == len[r]) continue;
                ssize_t k = recv(fd[r], from[r] + got[r], len[r] - got[r], MSG_DONTWAIT);
                if (k > 0) { got[r] += (int) k; if (got[r] == len[r]) --remaining; }
                else if (k == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) DIE("peer connection lost (read)");
            }
        }
    }

    // out: n * len bytes, rank order. in: this rank's payload.
    void allgather(const uint8_t * in, uint8_t * out, int len) {
        memcpy(out + (size_t) rank * len, in, len);
        if (n == 1) return;
        std::vector<const uint8_t *> to(n, in);
        std::vector<uint8_t *> from(n);
        std::vector<int> lens(n, len);
        for (int r = 0; r < n; ++r) from[r] = out + (size_t) r * len;
        round(to.data(), from.data(), lens.data());
    }

    // Rank 0 -> everyone. Used for requests in server mode.
    void bcast(uint8_t * buf, int len) {
        if (n == 1) return;
        if (rank == 0) { for (int r = 1; r < n; ++r) write_all(fd[r], buf, len); }
        else read_all(fd[0], buf, len);
    }

    // In-place all-reduce (sum). Rank r owns slice [r*len/n, (r+1)*len/n):
    // round 1 brings it every peer's partial of that slice, which it sums
    // in rank order (p0 + p1 + ... + p{n-1}, the same order on every rank
    // for every slice); round 2 puts the summed slices back in place
    // everywhere. Every rank ends with identical bits.
    void allreduce_sum(float * v, int len, std::vector<float> & scratch) {
        if (n == 1) return;
        std::vector<int> lo(n + 1);
        for (int r = 0; r <= n; ++r) lo[r] = (int) ((int64_t) len * r / n);
        const int mine = lo[rank + 1] - lo[rank];
        scratch.resize((size_t) n * mine);
        std::vector<const uint8_t *> to(n);
        std::vector<uint8_t *> from(n);
        std::vector<int> lens(n);
        for (int r = 0; r < n; ++r) { to[r] = (const uint8_t *) (v + lo[r]); from[r] = (uint8_t *) (scratch.data() + (size_t) r * mine); lens[r] = (lo[r + 1] - lo[r]) * 4; }
        round(to.data(), from.data(), lens.data());
        for (int i = 0; i < mine; ++i) {
            float s = 0;
            for (int r = 0; r < n; ++r) {
                const float p = r == rank ? v[lo[rank] + i] : scratch[(size_t) r * mine + i];
                s = r == 0 ? p : s + p;
            }
            v[lo[rank] + i] = s;
        }
        for (int r = 0; r < n; ++r) { to[r] = (const uint8_t *) (v + lo[rank]); from[r] = (uint8_t *) (v + lo[r]); lens[r] = (lo[r + 1] - lo[r]) * 4; }
        round(to.data(), from.data(), lens.data());
    }
};

// ------------------------------------------------------------------- ops

// Vector float32 exp, copied from ggml (ggml/src/ggml-cpu/vec.h, MIT), itself
// adapted from Arm's optimized routines: max error 1.45358 + 0.5 ulp.
static inline float32x4_t v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// out = x * rsqrt(mean(x^2) + eps) * w. The sum of squares accumulates in
// double (two float64 lanes), as the scalar version did. n % 4 == 0.
static void rmsnorm(float * out, const float * x, const float * w, int n, float eps) {
    float64x2_t s0 = vdupq_n_f64(0), s1 = vdupq_n_f64(0);
    for (int i = 0; i < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float64x2_t lo = vcvt_f64_f32(vget_low_f32(v)), hi = vcvt_high_f64_f32(v);
        s0 = vfmaq_f64(s0, lo, lo);
        s1 = vfmaq_f64(s1, hi, hi);
    }
    double ss = vaddvq_f64(vaddq_f64(s0, s1));
    float scale = 1.0f / sqrtf((float) (ss / n) + eps);
    const float32x4_t vs = vdupq_n_f32(scale);
    for (int i = 0; i < n; i += 4) vst1q_f32(out + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), vs), vld1q_f32(w + i)));
}

// LLaMA rope: adjacent pairs, theta_i = pos * base^(-2i/d) / freq_factor[i].
// The angles depend on the position only, so one table per token serves
// every layer's q and k.
static void rope_table(float * cs, int d, int pos, float base, const float * freqs) {
    for (int i = 0; i < d / 2; ++i) {
        float theta = pos * powf(base, -2.0f * i / d);
        if (freqs) theta /= freqs[i];
        cs[2 * i] = cosf(theta); cs[2 * i + 1] = sinf(theta);
    }
}
static void rope(float * x, int n_heads, int d, const float * cs) {
    for (int i = 0; i < d / 2; ++i) {
        const float c = cs[2 * i], s = cs[2 * i + 1];
        for (int h = 0; h < n_heads; ++h) {
            float * p = x + h * d + 2 * i;
            float x0 = p[0], x1 = p[1];
            p[0] = x0 * c - x1 * s;
            p[1] = x0 * s + x1 * c;
        }
    }
}

static void embed(float * out, const uint8_t * token_embd, int token, int n_embd) {
    const uint8_t * row = token_embd + (size_t) token * n_embd / 32 * 18;
    for (int b = 0; b < n_embd / 32; ++b) {
        const uint8_t * blk = row + b * 18;
        uint16_t dbits; memcpy(&dbits, blk, 2);
        float d = ggml_fp16_to_fp32(dbits);
        for (int j = 0; j < 16; ++j) {
            out[32 * b + j] = ((int) (blk[2 + j] & 0xF) - 8) * d;
            out[32 * b + j + 16] = ((int) (blk[2 + j] >> 4) - 8) * d;
        }
    }
}

// x += y, n % 4 == 0
static void add_into(float * x, const float * y, int n) {
    for (int i = 0; i < n; i += 4) vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(y + i)));
}

// act = silu(gate) * up, n % 4 == 0
static void silu_mul(float * act, const float * gate, const float * up, int n) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (int i = 0; i < n; i += 4) {
        float32x4_t g = vld1q_f32(gate + i);
        float32x4_t sg = vdivq_f32(g, vaddq_f32(one, v_expf(vnegq_f32(g))));
        vst1q_f32(act + i, vmulq_f32(sg, vld1q_f32(up + i)));
    }
}

// s[i] = exp(s[i] - mx) for i < n; returns the sum. The tail goes through
// the same vector exp so every element is rounded the same way.
static float softmax_exp(float * s, int n, float mx) {
    const float32x4_t vmx = vdupq_n_f32(mx);
    float32x4_t vsum = vdupq_n_f32(0);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t e = v_expf(vsubq_f32(vld1q_f32(s + i), vmx));
        vst1q_f32(s + i, e);
        vsum = vaddq_f32(vsum, e);
    }
    float sum = vaddvq_f32(vsum);
    for (; i < n; ++i) { s[i] = vgetq_lane_f32(v_expf(vdupq_n_f32(s[i] - mx)), 0); sum += s[i]; }
    return sum;
}

// Attention for one KV head and the B (<= 4) query heads that share it,
// over positions 0..pos. Each K row is loaded once and feeds B dot
// products; each V row is loaded once and feeds B outputs, 16 columns at a
// time so the accumulators stay in registers. Rows of the caches are
// `stride` floats apart; sc holds B rows of scores, sc_stride apart.
//   q: B heads of d floats   out: B heads of d floats   d % 16 == 0
template <int B>
static void attention_group(const float * q, const float * kc, const float * vc, size_t stride, int pos, int d,
                            float scale, float * sc, int sc_stride, float * out) {
    float mx[B];
    #pragma GCC unroll 4
    for (int b = 0; b < B; ++b) mx[b] = -1e30f;
    for (int t = 0; t <= pos; ++t) {
        const float * kt = kc + (size_t) t * stride;
        float32x4_t acc[B];
        #pragma GCC unroll 4
        for (int b = 0; b < B; ++b) acc[b] = vdupq_n_f32(0);
        for (int i = 0; i < d; i += 4) {
            const float32x4_t k4 = vld1q_f32(kt + i);
            #pragma GCC unroll 4
            for (int b = 0; b < B; ++b) acc[b] = vfmaq_f32(acc[b], k4, vld1q_f32(q + b * d + i));
        }
        #pragma GCC unroll 4
        for (int b = 0; b < B; ++b) {
            float sv = vaddvq_f32(acc[b]) * scale;
            sc[b * sc_stride + t] = sv;
            if (sv > mx[b]) mx[b] = sv;
        }
    }
    float inv[B];
    #pragma GCC unroll 4
    for (int b = 0; b < B; ++b) inv[b] = 1.0f / softmax_exp(sc + b * sc_stride, pos + 1, mx[b]);
    for (int i0 = 0; i0 < d; i0 += 16) {
        float32x4_t acc[B][4];
        #pragma GCC unroll 4
        for (int b = 0; b < B; ++b) {
            #pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) acc[b][j] = vdupq_n_f32(0);
        }
        for (int t = 0; t <= pos; ++t) {
            const float * vt = vc + (size_t) t * stride + i0;
            const float32x4_t v0 = vld1q_f32(vt), v1 = vld1q_f32(vt + 4), v2 = vld1q_f32(vt + 8), v3 = vld1q_f32(vt + 12);
            #pragma GCC unroll 4
            for (int b = 0; b < B; ++b) {
                const float32x4_t p = vdupq_n_f32(sc[b * sc_stride + t] * inv[b]);
                acc[b][0] = vfmaq_f32(acc[b][0], v0, p);
                acc[b][1] = vfmaq_f32(acc[b][1], v1, p);
                acc[b][2] = vfmaq_f32(acc[b][2], v2, p);
                acc[b][3] = vfmaq_f32(acc[b][3], v3, p);
            }
        }
        #pragma GCC unroll 4
        for (int b = 0; b < B; ++b) {
            #pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) vst1q_f32(out + b * d + i0 + 4 * j, acc[b][j]);
        }
    }
}

// Query heads [h0, h0 + n) all attend to the KV head whose cache rows start at kc / vc.
static void attention_heads(int n, const float * q, const float * kc, const float * vc, size_t stride, int pos, int d,
                            float scale, float * sc, int sc_stride, float * out) {
    switch (n) {
        case 4: attention_group<4>(q, kc, vc, stride, pos, d, scale, sc, sc_stride, out); break;
        case 3: attention_group<3>(q, kc, vc, stride, pos, d, scale, sc, sc_stride, out); break;
        case 2: attention_group<2>(q, kc, vc, stride, pos, d, scale, sc, sc_stride, out); break;
        default: attention_group<1>(q, kc, vc, stride, pos, d, scale, sc, sc_stride, out); break;
    }
}

// ------------------------------------------------------------------- stats

// What each board is doing, sampled once per token on every rank and carried
// to rank 0 inside the per-token candidate exchange (a few dozen bytes on a
// message that already exists). Rank 0 serves the latest set at /stats.
struct rank_stats {
    float engine_ms, comm_ms, cpu_ms;   // last token, this rank
    float tps;                          // this rank's view of the stream rate
    float cpu_pct;                      // all cores, since the previous sample
    float mem_used_mb, mem_total_mb;
    float temp_c;                       // hottest of the ZynqMP AMS sensors
    float power_w;                      // INA260 on the SOM, when present
};

struct stats_sampler {
    unsigned long long prev_busy = 0, prev_total = 0;
    double prev_t = 0, prev_engine = 0, prev_comm = 0, prev_cpu = 0;
    std::string temp_paths[3], power_path;
    void discover() {
        for (int h = 0; h < 8; ++h) {
            char nm[64], buf[32] = {0};
            snprintf(nm, sizeof nm, "/sys/class/hwmon/hwmon%d/name", h);
            FILE * f = fopen(nm, "r");
            if (!f) continue;
            if (fgets(buf, sizeof buf, f)) buf[strcspn(buf, "\n")] = 0;
            fclose(f);
            if (!strcmp(buf, "ams")) for (int i = 1; i <= 3; ++i) { snprintf(nm, sizeof nm, "/sys/class/hwmon/hwmon%d/temp%d_input", h, i); temp_paths[i - 1] = nm; }
            if (!strncmp(buf, "ina260", 6)) { snprintf(nm, sizeof nm, "/sys/class/hwmon/hwmon%d/power1_input", h); power_path = nm; }
        }
    }
    static long read_long(const std::string & path) {
        if (path.empty()) return -1;
        FILE * f = fopen(path.c_str(), "r");
        if (!f) return -1;
        long v = -1;
        if (fscanf(f, "%ld", &v) != 1) v = -1;
        fclose(f);
        return v;
    }
    rank_stats sample(double t_engine, double t_comm, double t_cpu, int tokens_so_far) {
        rank_stats st = {};
        double t = now();
        // per-token split: the deltas since the previous sample
        st.engine_ms = 1e3 * (t_engine - prev_engine); st.comm_ms = 1e3 * (t_comm - prev_comm); st.cpu_ms = 1e3 * (t_cpu - prev_cpu);
        st.tps = prev_t > 0 && t > prev_t ? (float) (1.0 / (t - prev_t)) : 0;
        prev_engine = t_engine; prev_comm = t_comm; prev_cpu = t_cpu; prev_t = t;
        // cpu: /proc/stat, all cores
        FILE * f = fopen("/proc/stat", "r");
        if (f) {
            unsigned long long u, n, sy, id, io, irq, sirq, steal;
            if (fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &u, &n, &sy, &id, &io, &irq, &sirq, &steal) == 8) {
                unsigned long long busy = u + n + sy + irq + sirq + steal, total = busy + id + io;
                if (prev_total && total > prev_total) st.cpu_pct = 100.0f * (busy - prev_busy) / (total - prev_total);
                prev_busy = busy; prev_total = total;
            }
            fclose(f);
        }
        f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[128]; long total = 0, avail = 0;
            while (fgets(line, sizeof line, f)) { sscanf(line, "MemTotal: %ld", &total); sscanf(line, "MemAvailable: %ld", &avail); }
            fclose(f);
            st.mem_total_mb = total / 1024.0f; st.mem_used_mb = (total - avail) / 1024.0f;
        }
        long tmax = -1;
        for (auto & tp : temp_paths) tmax = std::max(tmax, read_long(tp));
        st.temp_c = tmax > 0 ? tmax / 1000.0f : 0;
        long pw = read_long(power_path);
        st.power_w = pw > 0 ? pw / 1e6f : 0;
        (void) tokens_so_far;
        return st;
    }
};

// ------------------------------------------------------------------ decode

struct runtime {
    model_t M;
    engine_t E;
    comm_t C;
    int ctx_max;
    // KV cache: [layer][kv head][pos][head_dim] for keys and values, this
    // rank's heads only. One head's rows are contiguous so attention sweeps
    // two sequential streams; interleaving the heads per position (512-byte
    // stride, half of it used) cost 1.8x on the in-order A53.
    std::vector<float> kcache, vcache;
    std::vector<float> x, h, qkv, attn_out, partial, act, scratch, scores, rope_cs;
    double t_engine = 0, t_comm = 0, t_cpu = 0;
    stats_sampler sampler;
    std::vector<rank_stats> latest;      // rank 0: every rank's last sample
    std::vector<std::string> names;      // hostnames, rank order
    double latest_t = 0;
    int tokens_total = 0;
    // The last chat as a whole, for the page's system bar: every rank's
    // per-token split summed over the generation phase (the prefill sample
    // is left out), plus the request's own result. `chat` is the one in
    // progress; `last_chat` is published once it ends, under chat_mu since
    // /stats reads it from another thread.
    struct chat_stats { int n_prompt = 0, n_gen = 0, samples = 0; double prompt_s = 0, gen_s = 0; std::vector<double> engine_ms, comm_ms, cpu_ms; };
    chat_stats chat, last_chat;
    std::mutex chat_mu;

    void init(int ctx) {
        ctx_max = ctx;
        sampler.discover();
        latest.assign(M.nranks, rank_stats{});
        const hparams & hp = M.hp;
        kcache.assign((size_t) hp.n_layer * ctx * M.kv_rows, 0);
        vcache.assign((size_t) hp.n_layer * ctx * M.kv_rows, 0);
        if (hp.head_dim % 16 || hp.n_embd % 4 || M.ff_rows % 4) DIE("NEON ops need head_dim %% 16 == 0 and n_embd, ff_rows %% 4 == 0");
        x.resize(hp.n_embd); h.resize(hp.n_embd); attn_out.resize(hp.n_embd);
        partial.resize(hp.n_embd); act.resize(M.ff_rows); scores.resize((size_t) 4 * ctx); rope_cs.resize(hp.head_dim);
    }

    // Returns this rank's argmax over its vocab shard (global index, value).
    std::pair<int, float> forward(int token, int pos) {
        const hparams & hp = M.hp;
        const int heads_local = hp.n_head / M.nranks, kvh_local = hp.n_head_kv / M.nranks;
        const int d = hp.head_dim, group = hp.n_head / hp.n_head_kv;
        const float scale = 1.0f / sqrtf((float) d);
        double t0 = now();
        embed(x.data(), M.token_embd, token, hp.n_embd);
        rope_table(rope_cs.data(), d, pos, hp.rope_base, M.rope_freqs);

        for (int l = 0; l < hp.n_layer; ++l) {
            layer_w & L = M.layers[l];
            rmsnorm(h.data(), x.data(), L.attn_norm, hp.n_embd, hp.rms_eps);
            double t1 = now(); t_cpu += t1 - t0;
            E.gemv(L.qkv, h.data(), &L.o);                  // q | k | v shards
            t0 = now(); t_engine += t0 - t1;
            float * q = E.y, * k = E.y + M.q_rows, * v = E.y + M.q_rows + M.kv_rows;
            rope(q, heads_local, d, rope_cs.data());
            rope(k, kvh_local, d, rope_cs.data());
            for (int kvh = 0; kvh < kvh_local; ++kvh) {
                memcpy(&kcache[(((size_t) l * kvh_local + kvh) * ctx_max + pos) * d], k + kvh * d, d * 4);
                memcpy(&vcache[(((size_t) l * kvh_local + kvh) * ctx_max + pos) * d], v + kvh * d, d * 4);
            }
            // attention, this rank's heads; output goes to h (reused as o_in).
            // The heads of one KV group are done together, up to 4 at a time.
            for (int kvh = 0; kvh < kvh_local; ++kvh) {
                const float * kc0 = &kcache[((size_t) l * kvh_local + kvh) * ctx_max * d];
                const float * vc0 = &vcache[((size_t) l * kvh_local + kvh) * ctx_max * d];
                for (int g0 = 0; g0 < group; g0 += 4) {
                    const int hq = kvh * group + g0;
                    attention_heads(std::min(4, group - g0), q + hq * d, kc0, vc0, d, pos, d, scale,
                                    scores.data(), ctx_max, h.data() + hq * d);
                }
            }
            t1 = now(); t_cpu += t1 - t0;
            E.gemv(L.o, h.data(), &L.gateup);               // partial attn output [n_embd]
            t0 = now(); t_engine += t0 - t1;
            memcpy(partial.data(), E.y, hp.n_embd * 4);
            C.allreduce_sum(partial.data(), hp.n_embd, scratch);
            t1 = now(); t_comm += t1 - t0;
            add_into(x.data(), partial.data(), hp.n_embd);
            rmsnorm(h.data(), x.data(), L.ffn_norm, hp.n_embd, hp.rms_eps);
            t0 = now(); t_cpu += t0 - t1;
            E.gemv(L.gateup, h.data(), &L.down);            // gate | up shards
            t1 = now(); t_engine += t1 - t0;
            silu_mul(act.data(), E.y, E.y + M.ff_rows, M.ff_rows);
            t0 = now(); t_cpu += t0 - t1;
            E.gemv(L.down, act.data(), l + 1 < hp.n_layer ? &M.layers[l + 1].qkv : &M.lm_head);   // partial [n_embd]
            t1 = now(); t_engine += t1 - t0;
            memcpy(partial.data(), E.y, hp.n_embd * 4);
            C.allreduce_sum(partial.data(), hp.n_embd, scratch);
            t0 = now(); t_comm += t0 - t1;
            add_into(x.data(), partial.data(), hp.n_embd);
        }
        rmsnorm(h.data(), x.data(), M.output_norm, hp.n_embd, hp.rms_eps);
        double t1 = now(); t_cpu += t1 - t0;
        E.gemv(M.lm_head, h.data(), &M.layers[0].qkv);   // the next token starts with layer 0
        t0 = now(); t_engine += t0 - t1;
        int best = 0;
        for (int i = 1; i < M.vocab_rows; ++i) if (E.y[i] > E.y[best]) best = i;
        t_cpu += now() - t0;
        return {best + M.rank * M.vocab_rows, E.y[best]};
    }

    // Sampling parameters, identical on every rank (they travel with the request).
    struct sampling { float temperature = 0.0f; float top_p = 1.0f; int top_k = 40; uint32_t seed = 0; };

    // One token, chosen identically on every rank. Each rank contributes the
    // top_k logits of its vocab shard; every rank then runs the same
    // temperature / top-p sampler with the same seeded generator over the
    // union. Greedy (temperature 0) is the exact argmax; sampling is over
    // the n*top_k best candidates, which for top_k = 40 is all the mass a
    // top-p cut ever keeps.
    std::mt19937 rng;
    int sample_token(const sampling & sp) {
        struct cand { float v; int i; };
        const int k = std::max(1, std::min(sp.top_k, 64));
        // payload = k candidates + this rank's stats
        const int paylen = k * (int) sizeof(cand) + (int) sizeof(rank_stats);
        std::vector<uint8_t> mine_buf(paylen), all_buf((size_t) M.nranks * paylen);
        cand * mine_c = (cand *) mine_buf.data();
        std::vector<cand> mine(k, {-1e30f, 0});
        for (int i = 0; i < M.vocab_rows; ++i) {
            float v = E.y[i];
            if (v <= mine[k - 1].v) continue;
            int pos = k - 1;
            while (pos > 0 && mine[pos - 1].v < v) { mine[pos] = mine[pos - 1]; --pos; }
            mine[pos] = {v, i + M.rank * M.vocab_rows};
        }
        memcpy(mine_c, mine.data(), k * sizeof(cand));
        rank_stats st = sampler.sample(t_engine, t_comm, t_cpu, tokens_total);
        memcpy(mine_buf.data() + k * sizeof(cand), &st, sizeof st);
        double t0 = now();
        if (M.nranks > 1) C.allgather(mine_buf.data(), all_buf.data(), paylen);
        else memcpy(all_buf.data(), mine_buf.data(), paylen);
        t_comm += now() - t0;
        std::vector<cand> all((size_t) M.nranks * k);
        for (int r = 0; r < M.nranks; ++r) {
            memcpy(&all[(size_t) r * k], all_buf.data() + (size_t) r * paylen, k * sizeof(cand));
            memcpy(&latest[r], all_buf.data() + (size_t) r * paylen + k * sizeof(cand), sizeof(rank_stats));
        }
        latest_t = now();
        tokens_total++;
        std::sort(all.begin(), all.end(), [](const cand & a, const cand & b) { return a.v > b.v || (a.v == b.v && a.i < b.i); });
        if (sp.temperature <= 0.0f) return all[0].i;
        std::vector<float> p(all.size());
        float mx = all[0].v, sum = 0;
        for (size_t j = 0; j < all.size(); ++j) { p[j] = expf((all[j].v - mx) / sp.temperature); sum += p[j]; }
        float acc = 0, cut = sp.top_p * sum;
        size_t keep = all.size();
        for (size_t j = 0; j < all.size(); ++j) { acc += p[j]; if (acc >= cut) { keep = j + 1; break; } }
        std::uniform_real_distribution<float> u(0.0f, acc);
        float r = u(rng), c = 0;
        for (size_t j = 0; j < keep; ++j) { c += p[j]; if (r <= c) return all[j].i; }
        return all[keep - 1].i;
    }

    // Prefill the prompt, then generate up to n_max tokens. on_token is
    // called on every rank with each produced token (rank 0 streams it).
    // Returns the number of tokens produced and whether it stopped on EOG.
    struct result { int n_prompt, n_gen; bool eog; double prompt_s, gen_s; };
    result generate(const std::vector<llama_token> & prompt, int n_max, const sampling & sp,
                    const llama_vocab * vocab, const std::function<void(llama_token)> & on_token) {
        rng.seed(sp.seed);
        t_engine = t_comm = t_cpu = 0;
        chat = chat_stats{};
        chat.engine_ms.assign(M.nranks, 0); chat.comm_ms.assign(M.nranks, 0); chat.cpu_ms.assign(M.nranks, 0);
        double tp0 = now();
        for (size_t i = 0; i < prompt.size(); ++i) forward(prompt[i], (int) i);
        int tok = sample_token(sp);
        double tp1 = now();
        result res = {(int) prompt.size(), 0, false, tp1 - tp0, 0};
        for (int i = 0; i < n_max; ++i) {
            if (llama_vocab_is_eog(vocab, tok)) { res.eog = true; break; }
            on_token(tok);
            res.n_gen++;
            if (i + 1 == n_max || res.n_prompt + i + 1 >= ctx_max) break;
            forward(tok, res.n_prompt + i);
            tok = sample_token(sp);
            for (int r = 0; r < M.nranks; ++r) { chat.engine_ms[r] += latest[r].engine_ms; chat.comm_ms[r] += latest[r].comm_ms; chat.cpu_ms[r] += latest[r].cpu_ms; }
            chat.samples++;
        }
        res.gen_s = now() - tp1;
        chat.n_prompt = res.n_prompt; chat.n_gen = res.n_gen; chat.prompt_s = res.prompt_s; chat.gen_s = res.gen_s;
        { std::lock_guard<std::mutex> lk(chat_mu); last_chat = chat; }
        return res;
    }
};

// ------------------------------------------------------------------ server

// A request as it travels rank 0 -> peers: fixed header, then the tokens.
struct wire_request {
    uint32_t n_prompt, n_max, top_k, seed;
    float temperature, top_p;
};

// Holds back an incomplete trailing UTF-8 sequence so a streamed chunk is
// always valid text (a token can end mid-character).
struct utf8_stream {
    std::string pending;
    std::string push(const std::string & piece) {
        pending += piece;
        size_t n = pending.size(), cut = n;
        for (size_t back = 1; back <= 3 && back <= n; ++back) {
            unsigned char c = pending[n - back];
            if ((c & 0xC0) == 0x80) continue;              // continuation byte, keep looking
            int need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
            if (back < (size_t) need) cut = n - back;      // sequence not finished yet
            break;
        }
        std::string out = pending.substr(0, cut);
        pending.erase(0, cut);
        return out;
    }
};

static std::string token_piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof buf, 0, true);
    return n > 0 ? std::string(buf, n) : std::string();
}

static void serve(runtime & R, const llama_vocab * vocab, llama_model * lm, int http_port, const std::string & model_name) {
    const int rank = R.M.rank;
    std::vector<llama_token> toks;
    wire_request wr;

    // Peers: wait for requests from rank 0, run them, repeat.
    if (rank != 0) {
        for (;;) {
            R.C.bcast((uint8_t *) &wr, sizeof wr);
            toks.resize(wr.n_prompt);
            R.C.bcast((uint8_t *) toks.data(), (int) (wr.n_prompt * sizeof(llama_token)));
            runtime::sampling sp = {wr.temperature, wr.top_p, (int) wr.top_k, wr.seed};
            R.generate(toks, (int) wr.n_max, sp, vocab, [](llama_token) {});
        }
    }

    // Rank 0: the HTTP front. One request at a time -- the mesh is one
    // pipeline and the goal is the latency and rate of that one stream --
    // with a fair FIFO behind it: arrivals take a ticket and are served in
    // order, and a caller can see its place. More than QUEUE_MAX waiting is
    // refused with 429 rather than promised a slot minutes away.
    const int QUEUE_MAX = 8;
    std::mutex qmutex;
    std::condition_variable qcv;
    uint64_t next_ticket = 0, now_serving = 0;
    std::atomic<int> waiting{0};
    std::atomic<bool> generating{false};
    struct queue_ticket {
        uint64_t t; bool ok;
    };
    auto take_ticket = [&]() -> queue_ticket {
        std::lock_guard<std::mutex> l(qmutex);
        if (next_ticket - now_serving > (uint64_t) QUEUE_MAX) return {0, false};
        return {next_ticket++, true};
    };
    auto wait_turn = [&](uint64_t t) {
        std::unique_lock<std::mutex> l(qmutex);
        waiting++;
        qcv.wait(l, [&] { return now_serving == t; });
        waiting--;
    };
    auto release = [&]() {
        { std::lock_guard<std::mutex> l(qmutex); now_serving++; }
        qcv.notify_all();
    };
    const char * tmpl = llama_model_chat_template(lm, nullptr);
    const std::string chat_tmpl = tmpl ? tmpl : "llama3";
    httplib::Server srv;
    // Browsers: a page served from anywhere may call this API.
    srv.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
                             {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
    srv.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });

    auto run_request = [&](uint64_t ticket, const std::string & prompt, int n_max, runtime::sampling sp,
                           const std::function<void(const std::string &)> & emit) -> runtime::result {
        wait_turn(ticket);
        generating = true;
        struct done_t { std::function<void()> f; ~done_t() { f(); } } done{[&] { generating = false; release(); }};
        std::vector<llama_token> t(prompt.size() + 16);
        int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), t.data(), (int) t.size(), true, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), t.data(), (int) t.size(), true, true); }
        t.resize(std::max(n, 0));
        if ((int) t.size() >= R.ctx_max - 1) t.resize(R.ctx_max - 2);
        n_max = std::max(1, std::min(n_max, R.ctx_max - (int) t.size() - 1));
        wr = {(uint32_t) t.size(), (uint32_t) n_max, (uint32_t) sp.top_k, sp.seed, sp.temperature, sp.top_p};
        R.C.bcast((uint8_t *) &wr, sizeof wr);
        R.C.bcast((uint8_t *) t.data(), (int) (t.size() * sizeof(llama_token)));
        utf8_stream u8;
        runtime::result res = R.generate(t, n_max, sp, vocab, [&](llama_token tok) {
            std::string out = u8.push(token_piece(vocab, tok));
            if (!out.empty()) emit(out);
        });
        if (!u8.pending.empty()) emit(u8.pending);   // whatever is left, as-is
        return res;
    };

    auto sampling_from = [](const json & body) {
        runtime::sampling sp;
        sp.temperature = body.value("temperature", 0.0f);
        sp.top_p = body.value("top_p", 1.0f);
        sp.top_k = body.value("top_k", 40);
        sp.seed = body.value("seed", (uint32_t) time(nullptr));
        return sp;
    };
    auto timings = [](const runtime::result & r) {
        return json{{"prompt_n", r.n_prompt}, {"prompt_ms", 1e3 * r.prompt_s},
                    {"predicted_n", r.n_gen}, {"predicted_ms", 1e3 * r.gen_s},
                    {"predicted_per_second", r.n_gen / std::max(r.gen_s, 1e-9)}};
    };
    auto sse = [](const json & j) { return "data: " + j.dump() + "\n\n"; };

    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"object", "list"}, {"data", {{{"id", model_name}, {"object", "model"}, {"owned_by", "llm-kv260"}}}}}.dump(), "application/json");
    });
    // shard size: what this configuration keeps in each board's engine memory
    size_t shard_bytes = R.M.lm_head.bytes();
    for (auto & L : R.M.layers) shard_bytes += L.qkv.bytes() + L.o.bytes() + L.gateup.bytes() + L.down.bytes();
    srv.Get("/stats", [&](const httplib::Request &, httplib::Response & res) {
        json boards = json::array();
        const bool busy = generating.load();
        const double age = R.latest_t > 0 ? now() - R.latest_t : -1;
        for (int r = 0; r < R.M.nranks; ++r) {
            const rank_stats & st = R.latest[r];
            const float tok_ms = st.engine_ms + st.comm_ms + st.cpu_ms;
            boards.push_back({{"rank", r}, {"host", r < (int) R.names.size() ? R.names[r] : ""},
                              {"state", busy ? "decoding" : "idle"},
                              {"engine_ms", st.engine_ms}, {"exchange_ms", st.comm_ms}, {"cpu_ms", st.cpu_ms},
                              {"engine_share", tok_ms > 0 ? st.engine_ms / tok_ms : 0},
                              {"tps", st.tps}, {"cpu_pct", st.cpu_pct},
                              {"mem_used_mb", st.mem_used_mb}, {"mem_total_mb", st.mem_total_mb},
                              {"temp_c", st.temp_c}, {"power_w", st.power_w},
                              {"shard_mb", shard_bytes / 1e6},
                              {"heads", std::to_string(r * R.M.hp.n_head / R.M.nranks) + "-" + std::to_string((r + 1) * R.M.hp.n_head / R.M.nranks - 1)},
                              {"vocab_rows", std::to_string(r * R.M.vocab_rows) + "-" + std::to_string((r + 1) * R.M.vocab_rows - 1)}});
        }
        // The last chat: per-token averages over its generation phase, per rank and as the mean across ranks.
        json last;
        {
            std::lock_guard<std::mutex> lk(R.chat_mu);
            const runtime::chat_stats & c = R.last_chat;
            const double n = std::max(c.samples, 1), nr = std::max((int) c.engine_ms.size(), 1);
            json per = json::array();
            double e = 0, x = 0, cpu = 0;
            for (int r = 0; r < (int) c.engine_ms.size(); ++r) {
                per.push_back({{"rank", r}, {"engine_ms", c.engine_ms[r] / n}, {"exchange_ms", c.comm_ms[r] / n}, {"cpu_ms", c.cpu_ms[r] / n}});
                e += c.engine_ms[r] / n; x += c.comm_ms[r] / n; cpu += c.cpu_ms[r] / n;
            }
            last = {{"n_prompt", c.n_prompt}, {"n_gen", c.n_gen}, {"samples", c.samples},
                    {"prompt_ms", 1e3 * c.prompt_s}, {"gen_ms", 1e3 * c.gen_s}, {"tps", c.n_gen / std::max(c.gen_s, 1e-9)},
                    {"engine_ms", e / nr}, {"exchange_ms", x / nr}, {"cpu_ms", cpu / nr}, {"boards", per}};
        }
        res.set_content(json{{"busy", busy}, {"queued", waiting.load()}, {"model", model_name}, {"nranks", R.M.nranks},
                             {"sample_age_s", age}, {"tokens_total", R.tokens_total}, {"boards", boards}, {"last_chat", last}}.dump(), "application/json");
    });
    srv.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"status", "ok"}, {"ranks", R.M.nranks}, {"busy", generating.load()},
                             {"queued", waiting.load()}, {"queue_max", QUEUE_MAX}}.dump(), "application/json");
    });

    // Both endpoints share one implementation; only the prompt source and the JSON shape differ.
    auto handle = [&](const httplib::Request & req, httplib::Response & res, bool chat) {
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content(json{{"error", {{"message", "invalid JSON"}}}}.dump(), "application/json"); return;
        }
        std::string prompt;
        if (chat) {
            std::vector<llama_chat_message> msgs;
            std::vector<std::string> hold;
            for (auto & m : body.value("messages", json::array())) {
                hold.push_back(m.value("role", "user")); hold.push_back(m.value("content", ""));
            }
            for (size_t i = 0; i + 1 < hold.size(); i += 2) msgs.push_back({hold[i].c_str(), hold[i + 1].c_str()});
            std::vector<char> buf(4096);
            int n = llama_chat_apply_template(chat_tmpl.c_str(), msgs.data(), msgs.size(), true, buf.data(), (int) buf.size());
            if (n > (int) buf.size()) { buf.resize(n); n = llama_chat_apply_template(chat_tmpl.c_str(), msgs.data(), msgs.size(), true, buf.data(), (int) buf.size()); }
            if (n < 0) { res.status = 400; res.set_content(json{{"error", {{"message", "chat template failed"}}}}.dump(), "application/json"); return; }
            prompt.assign(buf.data(), n);
        } else {
            prompt = body.value("prompt", "");
        }
        const int n_max = body.value("max_tokens", 128);
        const bool stream = body.value("stream", false);
        const runtime::sampling sp = sampling_from(body);
        const std::string id = std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string((long) time(nullptr));
        const long created = (long) time(nullptr);
        const char * object = chat ? "chat.completion" : "text_completion";
        const queue_ticket tk = take_ticket();
        if (!tk.ok) {
            res.status = 429;
            res.set_content(json{{"error", {{"message", "queue full: one stream at a time, " + std::to_string(QUEUE_MAX) + " may wait"}, {"type", "busy"}}}}.dump(), "application/json");
            return;
        }
        const uint64_t ticket = tk.t;

        if (!stream) {
            std::string text;
            runtime::result r = run_request(ticket, prompt, n_max, sp, [&](const std::string & s) { text += s; });
            json choice = chat
                ? json{{"index", 0}, {"message", {{"role", "assistant"}, {"content", text}}}, {"finish_reason", r.eog ? "stop" : "length"}}
                : json{{"index", 0}, {"text", text}, {"finish_reason", r.eog ? "stop" : "length"}};
            json out = {{"id", id}, {"object", object}, {"created", created}, {"model", model_name},
                        {"choices", {choice}},
                        {"usage", {{"prompt_tokens", r.n_prompt}, {"completion_tokens", r.n_gen}, {"total_tokens", r.n_prompt + r.n_gen}}},
                        {"timings", timings(r)}};
            res.set_content(out.dump(), "application/json");
            return;
        }
        // Streaming: the whole generation runs inside the content provider.
        res.set_chunked_content_provider("text/event-stream", [=, &run_request, &sse](size_t, httplib::DataSink & sink) {
            // Tell a queued client where it stands: an SSE comment line per
            // second until its turn (clients ignore comments; a human sees
            // the page's "queued" note).
            auto chunk = [&](const json & delta_or_text, const char * finish) {
                json choice = chat
                    ? json{{"index", 0}, {"delta", delta_or_text}, {"finish_reason", finish ? json(finish) : json(nullptr)}}
                    : json{{"index", 0}, {"text", delta_or_text.value("content", "")}, {"finish_reason", finish ? json(finish) : json(nullptr)}};
                json j = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created},
                          {"model", model_name}, {"choices", {choice}}};
                std::string s = sse(j);
                sink.write(s.data(), s.size());
            };
            if (chat) chunk(json{{"role", "assistant"}, {"content", ""}}, nullptr);
            runtime::result r = run_request(ticket, prompt, n_max, sp, [&](const std::string & s) { chunk(json{{"content", s}}, nullptr); });
            json last = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created}, {"model", model_name},
                         {"choices", {chat ? json{{"index", 0}, {"delta", json::object()}, {"finish_reason", r.eog ? "stop" : "length"}}
                                           : json{{"index", 0}, {"text", ""}, {"finish_reason", r.eog ? "stop" : "length"}}}},
                         {"usage", {{"prompt_tokens", r.n_prompt}, {"completion_tokens", r.n_gen}, {"total_tokens", r.n_prompt + r.n_gen}}},
                         {"timings", timings(r)}};
            std::string s = sse(last) + "data: [DONE]\n\n";
            sink.write(s.data(), s.size());
            sink.done();
            return true;
        });
    };
    srv.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) { handle(req, res, true); });
    srv.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) { handle(req, res, false); });

    fprintf(stderr, "rank 0: serving OpenAI-compatible API on port %d, %d rank(s), model %s\n", http_port, R.M.nranks, model_name.c_str());
    srv.listen("0.0.0.0", http_port);
}

// -------------------------------------------------------------------- main

int main(int argc, char ** argv) {
    std::string model_path, prompt = "The capital of France is", peers_arg;
    int rank = 0, nranks = 1, n_gen = 32, port = 5555, ctx = 2048, serve_port = 0, cpu = 2;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() { if (i + 1 >= argc) DIE("%s needs a value", a.c_str()); return std::string(argv[++i]); };
        if (a == "--model") model_path = val();
        else if (a == "--rank") rank = atoi(val().c_str());
        else if (a == "--nranks") nranks = atoi(val().c_str());
        else if (a == "--peers") peers_arg = val();
        else if (a == "--prompt") prompt = val();
        else if (a == "--n") n_gen = atoi(val().c_str());
        else if (a == "--port") port = atoi(val().c_str());
        else if (a == "--ctx") ctx = atoi(val().c_str());
        else if (a == "--cpu") cpu = atoi(val().c_str());
        else if (a == "--serve") serve_port = atoi(val().c_str());
        else DIE("unknown arg %s", a.c_str());
    }
    if (model_path.empty()) DIE("--model required");
    std::vector<std::string> peers;
    if (nranks > 1) {
        size_t p = 0;
        while (p <= peers_arg.size()) {
            size_t c = peers_arg.find(',', p);
            peers.push_back(peers_arg.substr(p, c == std::string::npos ? std::string::npos : c - p));
            if (c == std::string::npos) break;
            p = c + 1;
        }
        if ((int) peers.size() != nranks) DIE("--peers must list %d addresses", nranks);
    } else {
        peers.push_back("127.0.0.1");
    }

    // One core for the whole process (--cpu -1 to leave it to the scheduler).
    // CPU 0 takes the DisplayPort interrupt storm and CPU 3 the NIC's IRQ;
    // on CPU 2 the exchange floor is 30% lower and its tail half (0011).
    if (cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof set, &set)) fprintf(stderr, "rank %d: could not pin to cpu %d\n", rank, cpu);
    }

    runtime R;
    R.M.rank = rank; R.M.nranks = nranks;
    pools_t pools; pools.open();
    gguf_file f; f.open(model_path.c_str());
    double t0 = now();
    R.E.probe();   // which engine the overlay provides decides the weight layout
    load_model(R.M, f, pools, R.E.v2);
    R.E.open(pools, std::max(R.M.vocab_rows, 2 * R.M.ff_rows));
    R.C.open(rank, peers, port);
    R.init(ctx);
    fprintf(stderr, "rank %d: loaded in %.1f s\n", rank, now() - t0);

    // Tokenizer: llama.cpp's, vocab only. Every rank tokenizes identically.
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * lm = llama_model_load_from_file(model_path.c_str(), mp);
    if (!lm) DIE("vocab load failed");
    const llama_vocab * vocab = llama_model_get_vocab(lm);
    // The connection mesh is the barrier; the first exchange carries hostnames.
    {
        char host[32] = {0};
        gethostname(host, sizeof host - 1);
        std::vector<uint8_t> all((size_t) nranks * 32);
        R.C.allgather((const uint8_t *) host, all.data(), 32);
        for (int r = 0; r < nranks; ++r) R.names.push_back(std::string((const char *) &all[(size_t) r * 32]));
        // A first stats sample from every board, so the page shows real
        // resource numbers before the first token is ever generated.
        rank_stats st = R.sampler.sample(0, 0, 0, 0);
        st.tps = 0;
        std::vector<uint8_t> sall((size_t) nranks * sizeof(rank_stats));
        R.C.allgather((const uint8_t *) &st, sall.data(), sizeof st);
        for (int r = 0; r < nranks; ++r) memcpy(&R.latest[r], &sall[(size_t) r * sizeof(rank_stats)], sizeof(rank_stats));
        R.latest_t = now();
    }

    if (serve_port) {
        std::string name = model_path.substr(model_path.find_last_of('/') + 1);
        serve(R, vocab, lm, serve_port, name);   // never returns
    }

    std::vector<llama_token> toks(prompt.size() + 8);
    int n_prompt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, false);
    if (n_prompt < 0) DIE("tokenize failed");
    toks.resize(n_prompt);
    if (n_prompt + n_gen > ctx) DIE("prompt + n exceeds --ctx %d", ctx);

    runtime::sampling greedy;
    runtime::result r = R.generate(toks, n_gen, greedy, vocab, [&](llama_token tok) {
        if (rank == 0) { std::string s = token_piece(vocab, tok); fwrite(s.data(), 1, s.size(), stdout); fflush(stdout); }
    });
    if (rank == 0) printf("\n");
    fprintf(stderr, "rank %d: prompt %d tokens in %.2f s (%.1f t/s); generated %d tokens in %.2f s = %.2f t/s; per token: engine %.1f ms, comm %.1f ms, cpu %.1f ms\n",
            rank, r.n_prompt, r.prompt_s, r.n_prompt / r.prompt_s, r.n_gen, r.gen_s, r.n_gen / r.gen_s,
            1e3 * R.t_engine / std::max(r.n_gen, 1), 1e3 * R.t_comm / std::max(r.n_gen, 1), 1e3 * R.t_cpu / std::max(r.n_gen, 1));
    llama_model_free(lm);
    return 0;
}
