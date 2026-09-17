// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of w0
//        bit 31~0 - w0[31:0] (Read/Write)
// 0x14 : Data signal of w0
//        bit 31~0 - w0[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of w1
//        bit 31~0 - w1[31:0] (Read/Write)
// 0x20 : Data signal of w1
//        bit 31~0 - w1[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of w2
//        bit 31~0 - w2[31:0] (Read/Write)
// 0x2c : Data signal of w2
//        bit 31~0 - w2[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of w3
//        bit 31~0 - w3[31:0] (Read/Write)
// 0x38 : Data signal of w3
//        bit 31~0 - w3[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of beats_per_port
//        bit 31~0 - beats_per_port[31:0] (Read/Write)
// 0x44 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XGEMV_FETCH_CONTROL_ADDR_AP_CTRL             0x00
#define XGEMV_FETCH_CONTROL_ADDR_GIE                 0x04
#define XGEMV_FETCH_CONTROL_ADDR_IER                 0x08
#define XGEMV_FETCH_CONTROL_ADDR_ISR                 0x0c
#define XGEMV_FETCH_CONTROL_ADDR_W0_DATA             0x10
#define XGEMV_FETCH_CONTROL_BITS_W0_DATA             64
#define XGEMV_FETCH_CONTROL_ADDR_W1_DATA             0x1c
#define XGEMV_FETCH_CONTROL_BITS_W1_DATA             64
#define XGEMV_FETCH_CONTROL_ADDR_W2_DATA             0x28
#define XGEMV_FETCH_CONTROL_BITS_W2_DATA             64
#define XGEMV_FETCH_CONTROL_ADDR_W3_DATA             0x34
#define XGEMV_FETCH_CONTROL_BITS_W3_DATA             64
#define XGEMV_FETCH_CONTROL_ADDR_BEATS_PER_PORT_DATA 0x40
#define XGEMV_FETCH_CONTROL_BITS_BEATS_PER_PORT_DATA 32

