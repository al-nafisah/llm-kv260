// Types and block layout for y = W x with W in ggml Q4_0 and x in Q8_0,
// shared by the kernels in hls/gemv2/ and the MAC in gemv_core.h.
//
// A kernel built on these streams rows straight out of the model's own bytes
// in DRAM and computes them in the order src/llm_kv260/q4_0.py pins. That
// module is the contract; gemv_core.h is held to it, byte for byte.
#pragma once

#define AP_INT_MAX_W 2048  // the unpacker holds a 9-beat, 1152-bit chunk
#include <ap_int.h>

typedef ap_uint<128> beat_t;   // one AXI beat: the HP ports are 128 bits wide
typedef ap_uint<144> block_t;  // one Q4_0 block: float16 d, then 16 nibble bytes

static const int QK = 32;                   // weights per block
static const int BLOCK_BYTES = 18;
static const int MAX_K = 8192;              // Llama-3.2-1B down_proj
static const int MAX_NB = MAX_K / QK;       // 256 blocks per row
static const int LANES = 8;

// Layout arithmetic. A row is K/32 blocks of 18 bytes; 8 blocks are 144 bytes
// = exactly 9 beats, so a row of K (multiple of 256) inputs is K/256 chunks
// of 9 beats and the unpacker never sees a block straddle a chunk.
static const int BEATS_PER_CHUNK = 9;
static const int BLOCKS_PER_CHUNK = 8;
