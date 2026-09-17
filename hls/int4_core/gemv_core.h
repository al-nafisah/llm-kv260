// The datapath every MAC unit in hls/gemv2/ runs. The arithmetic is the
// contract's (src/llm_kv260/q4_0.py) and must not change: one differing byte
// between fabric and contract is a failure.
#pragma once

#include "q4_0.h"

#include <hls_stream.h>

typedef ap_uint<256> xblock_t;  // 32 int8 activations of one block

// ----------------------------------------------------------------- unpack

static void unpack(const beat_t *w, int beats, hls::stream<block_t> &out) {
    // Block j of a 9-beat chunk occupies bytes 18j..18j+17, which always lie
    // inside beats j and j+1 at byte offset 2j. So a two-beat window and an
    // 8-way mux are enough; no wide buffer, no variable-position write.
    beat_t prev = 0;
    ap_uint<4> i = 0;  // beat index within the chunk
unpack_loop:
    for (int t = 0; t < beats; ++t) {
#pragma HLS PIPELINE II = 1
        beat_t cur = w[t];
        if (i >= 1) {
            ap_uint<256> window = (ap_uint<256>(cur) << 128) | ap_uint<256>(prev);
            ap_uint<3> j = i - 1;
            block_t blk;
            switch (j) {
                case 0: blk = window.range(16 * 0 + 143, 16 * 0); break;
                case 1: blk = window.range(16 * 1 + 143, 16 * 1); break;
                case 2: blk = window.range(16 * 2 + 143, 16 * 2); break;
                case 3: blk = window.range(16 * 3 + 143, 16 * 3); break;
                case 4: blk = window.range(16 * 4 + 143, 16 * 4); break;
                case 5: blk = window.range(16 * 5 + 143, 16 * 5); break;
                case 6: blk = window.range(16 * 6 + 143, 16 * 6); break;
                default: blk = window.range(16 * 7 + 143, 16 * 7); break;
            }
            out.write(blk);
        }
        prev = cur;
        i = (i == BEATS_PER_CHUNK - 1) ? ap_uint<4>(0) : ap_uint<4>(i + 1);
    }
}

// ------------------------------------------------------------ half -> float

// Exact by construction (every float16 is a float32), so it is pure wiring
// plus a leading-zero count for subnormals. Written out rather than taken
// from hls_half.h so the design carries no floating-point IP for it and the
// C testbench links against nothing but libc.
static float half_to_float(ap_uint<16> h) {
#pragma HLS INLINE
    ap_uint<1> sign = h[15];
    ap_uint<5> exp = h.range(14, 10);
    ap_uint<10> man = h.range(9, 0);
    ap_uint<32> bits = 0;
    if (exp == 0) {
        if (man != 0) {
            // Subnormal: value = man * 2^-24. Normalise to 1.xxx * 2^(p-24)
            // where p is the leading one's position.
            ap_uint<4> p = 0;
        clz:
            for (int i = 9; i >= 0; --i) {
#pragma HLS UNROLL
                if (man[i]) { p = i; break; }
            }
            ap_uint<10> shifted = man << (9 - p);        // leading one now at bit 9
            ap_uint<10> frac = shifted.range(8, 0) << 1; // drop it, left-align the rest
            bits.range(22, 13) = frac;
            bits.range(30, 23) = ap_uint<8>(p + 103);    // (p - 24) + 127
        }
    } else if (exp == 31) {
        bits.range(30, 23) = 0xFF;
        bits.range(22, 13) = man;
    } else {
        bits.range(30, 23) = ap_uint<8>(exp + 112);      // rebias 15 -> 127
        bits.range(22, 13) = man;
    }
    bits[31] = sign;
    return *reinterpret_cast<float *>(&bits);
}

// -------------------------------------------------------------------- mac

static float tree8(const float l[LANES]) {
#pragma HLS INLINE
    // The contract's tree_sum: adjacent pairs, three levels.
    return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
}

static void mac(hls::stream<block_t> &in, const xblock_t xq[MAX_NB], const float xd[MAX_NB],
                int nb, int rows, beat_t *y) {
    // Sixteen lanes in a small two-port RAM: eight for the row being
    // accumulated, eight holding the previous row. Lane l of a row is
    // written once every eight blocks, so a read of it is always at least
    // eight iterations after the previous write, which is what lets a
    // multi-cycle float adder sit inside a one-block-per-cycle loop. The
    // DEPENDENCE pragma states exactly that distance.
    //
    // The tree over the previous row is read out one lane per cycle during
    // the first eight blocks of the next row, when the accumulator itself
    // reads nothing (those blocks start from 0.0f), so one read port serves
    // both. Lane l of the previous row was last written exactly eight
    // iterations before it is read here.
    float lanes[2 * LANES];
#pragma HLS BIND_STORAGE variable = lanes type = ram_s2p impl = lutram
    float tree_in[LANES];
#pragma HLS ARRAY_PARTITION variable = tree_in complete
    ap_uint<16> b = 0;       // block within row (or within the 8-cycle epilogue)
    ap_uint<1> s = 0;        // lane set of the current row
    int rows_done = 0;
    ap_uint<2> r4 = 0;       // row within the 4-row output beat
    beat_t ybeat = 0;
    int ybeats = 0;
    const int total = rows * nb;

mac_loop:
    for (int t = 0; t < total + LANES; ++t) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = lanes inter distance = 8 true
        const bool active = t < total;
        const bool first8 = b < LANES;
        const ap_uint<3> lane = b.range(2, 0);

        // The one RAM read this iteration: this row's lane (to accumulate
        // into) after the first eight blocks, the previous row's lane
        // (for the tree) during them.
        const ap_uint<4> rd_addr = first8 ? ap_uint<4>((ap_uint<1>(!s), lane)) : ap_uint<4>((s, lane));
        const float rd = lanes[rd_addr];

        if (active) {
            block_t blk = in.read();

            // Step 1: int32 block dot product. Nibble n is weight n-8; byte j
            // carries weight j (low) and weight j+16 (high).
            ap_uint<16> dbits = blk.range(15, 0);
            ap_uint<128> nib = blk.range(143, 16);
            xblock_t xb = xq[b];
            ap_int<16> sumi = 0;
        dot:
            for (int j = 0; j < QK / 2; ++j) {
#pragma HLS UNROLL
                ap_int<5> lo = ap_int<5>(ap_uint<4>(nib.range(8 * j + 3, 8 * j))) - 8;
                ap_int<5> hi = ap_int<5>(ap_uint<4>(nib.range(8 * j + 7, 8 * j + 4))) - 8;
                ap_int<8> xlo = xb.range(8 * j + 7, 8 * j);
                ap_int<8> xhi = xb.range(8 * (j + 16) + 7, 8 * (j + 16));
                sumi += lo * xlo + hi * xhi;
            }

            // Step 2: two float32 multiplies, left to right. half -> float is exact.
            float t_blk = ((float)(int)sumi * half_to_float(dbits)) * xd[b];

            // Step 3: lane b mod 8. The first eight blocks start from
            // 0.0f + t, as the contract does, rather than from t.
            float prev = first8 ? 0.0f : rd;
            lanes[ap_uint<4>((s, lane))] = prev + t_blk;
        }

        // Step 4: collect the previous row's lanes, then the tree.
        if (first8) {
            tree_in[lane] = rd;
        }
        if (b == LANES - 1 && rows_done > 0) {
            float yv = tree8(tree_in);
            ap_uint<32> ybits = *reinterpret_cast<ap_uint<32> *>(&yv);
            ybeat.range(32 * r4 + 31, 32 * r4) = ybits;
            if (r4 == 3) {
                y[ybeats++] = ybeat;
            }
            r4++;
        }

        if (active && b == nb - 1) {
            b = 0;
            s = !s;
            rows_done++;
        } else {
            b++;
        }
    }
}

static void unpack_stream(hls::stream<beat_t> &w, int beats, hls::stream<block_t> &out) {
    // The same window, fed from a stream (gemv2's MAC kernel).
    beat_t prev = 0;
    ap_uint<4> i = 0;  // beat index within the chunk
unpack_stream_loop:
    for (int t = 0; t < beats; ++t) {
#pragma HLS PIPELINE II = 1
        beat_t cur = w.read();
        if (i >= 1) {
            ap_uint<256> window = (ap_uint<256>(cur) << 128) | ap_uint<256>(prev);
            ap_uint<3> j = i - 1;
            block_t blk;
            switch (j) {
                case 0: blk = window.range(16 * 0 + 143, 16 * 0); break;
                case 1: blk = window.range(16 * 1 + 143, 16 * 1); break;
                case 2: blk = window.range(16 * 2 + 143, 16 * 2); break;
                case 3: blk = window.range(16 * 3 + 143, 16 * 3); break;
                case 4: blk = window.range(16 * 4 + 143, 16 * 4); break;
                case 5: blk = window.range(16 * 5 + 143, 16 * 5); break;
                case 6: blk = window.range(16 * 6 + 143, 16 * 6); break;
                default: blk = window.range(16 * 7 + 143, 16 * 7); break;
            }
            out.write(blk);
        }
        prev = cur;
        i = (i == BEATS_PER_CHUNK - 1) ? ap_uint<4>(0) : ap_uint<4>(i + 1);
    }
}

