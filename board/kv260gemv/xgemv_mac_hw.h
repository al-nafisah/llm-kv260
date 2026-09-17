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
// 0x10 : Data signal of x
//        bit 31~0 - x[31:0] (Read/Write)
// 0x14 : Data signal of x
//        bit 31~0 - x[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of y0a
//        bit 31~0 - y0a[31:0] (Read/Write)
// 0x20 : Data signal of y0a
//        bit 31~0 - y0a[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of y0b
//        bit 31~0 - y0b[31:0] (Read/Write)
// 0x2c : Data signal of y0b
//        bit 31~0 - y0b[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of y1a
//        bit 31~0 - y1a[31:0] (Read/Write)
// 0x38 : Data signal of y1a
//        bit 31~0 - y1a[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of y1b
//        bit 31~0 - y1b[31:0] (Read/Write)
// 0x44 : Data signal of y1b
//        bit 31~0 - y1b[63:32] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of y2a
//        bit 31~0 - y2a[31:0] (Read/Write)
// 0x50 : Data signal of y2a
//        bit 31~0 - y2a[63:32] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of y2b
//        bit 31~0 - y2b[31:0] (Read/Write)
// 0x5c : Data signal of y2b
//        bit 31~0 - y2b[63:32] (Read/Write)
// 0x60 : reserved
// 0x64 : Data signal of y3a
//        bit 31~0 - y3a[31:0] (Read/Write)
// 0x68 : Data signal of y3a
//        bit 31~0 - y3a[63:32] (Read/Write)
// 0x6c : reserved
// 0x70 : Data signal of y3b
//        bit 31~0 - y3b[31:0] (Read/Write)
// 0x74 : Data signal of y3b
//        bit 31~0 - y3b[63:32] (Read/Write)
// 0x78 : reserved
// 0x7c : Data signal of k
//        bit 31~0 - k[31:0] (Read/Write)
// 0x80 : reserved
// 0x84 : Data signal of rows_per_mac
//        bit 31~0 - rows_per_mac[31:0] (Read/Write)
// 0x88 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XGEMV_MAC_CONTROL_ADDR_AP_CTRL           0x00
#define XGEMV_MAC_CONTROL_ADDR_GIE               0x04
#define XGEMV_MAC_CONTROL_ADDR_IER               0x08
#define XGEMV_MAC_CONTROL_ADDR_ISR               0x0c
#define XGEMV_MAC_CONTROL_ADDR_X_DATA            0x10
#define XGEMV_MAC_CONTROL_BITS_X_DATA            64
#define XGEMV_MAC_CONTROL_ADDR_Y0A_DATA          0x1c
#define XGEMV_MAC_CONTROL_BITS_Y0A_DATA          64
#define XGEMV_MAC_CONTROL_ADDR_Y0B_DATA          0x28
#define XGEMV_MAC_CONTROL_BITS_Y0B_DATA          64
#define XGEMV_MAC_CONTROL_ADDR_Y1A_DATA          0x34
#define XGEMV_MAC_CONTROL_BITS_Y1A_DATA          64
#define XGEMV_MAC_CONTROL_ADDR_Y1B_DATA          0x40
#define XGEMV_MAC_CONTROL_BITS_Y1B_DATA          64
#define XGEMV_MAC_CONTROL_ADDR_Y2A_DATA          0x4c
#define XGEMV_MAC_CONTROL_BITS_Y2A_DATA          64
#define XGEMV_MAC_CONTROL_ADDR_Y2B_DATA          0x58
#define XGEMV_MAC_CONTROL_BITS_Y2B_DATA          64
#define XGEMV_MAC_CONTROL_ADDR_Y3A_DATA          0x64
#define XGEMV_MAC_CONTROL_BITS_Y3A_DATA          64
#define XGEMV_MAC_CONTROL_ADDR_Y3B_DATA          0x70
#define XGEMV_MAC_CONTROL_BITS_Y3B_DATA          64
#define XGEMV_MAC_CONTROL_ADDR_K_DATA            0x7c
#define XGEMV_MAC_CONTROL_BITS_K_DATA            32
#define XGEMV_MAC_CONTROL_ADDR_ROWS_PER_MAC_DATA 0x84
#define XGEMV_MAC_CONTROL_BITS_ROWS_PER_MAC_DATA 32

