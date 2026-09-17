#!/bin/bash
# Runs on a board after `kit.tgz` (see docs/kv260-bringup.md) lands in ~:
# unpack, build u-dma-buf and the bench, compile the overlay, load it, and
# prove the engine is bit-exact. Idempotent.
#
# Loading the overlay needs root. Boards 1 and 2 have passwordless sudo; on
# the others run this as `SUDO_PASS=<password> bash ~/board-setup.sh`.
set -e
cd ~ && rm -rf kit && tar xzf kit.tgz
rm -rf ~/board ~/gemv2 ~/vectors ~/kv260gemv && mv kit/board ~/board && mv kit/gemv2 ~/gemv2 && mv kit/vectors ~/vectors && cp -r ~/board/kv260gemv ~/kv260gemv
rm -rf ~/udmabuf && tar xzf kit/udmabuf-src.tgz && (cd ~/udmabuf && make KERNEL_SRC=/usr/src/linux-headers-$(uname -r) > build.log 2>&1) && ls -l ~/udmabuf/u-dma-buf.ko | awk '{print "u-dma-buf.ko", $5}'
(cd ~/kv260gemv && make > /dev/null 2>&1) && echo "bench_gemv2 built"
(cd ~/gemv2 && dtc -@ -I dts -O dtb -o gemv2.dtbo pl.dtsi 2>/dev/null) && echo "dtbo compiled"
if [ -n "${SUDO_PASS:-}" ]; then echo "$SUDO_PASS" | sudo -S bash ~/board/load-overlay.sh ~/gemv2; else sudo bash ~/board/load-overlay.sh ~/gemv2; fi 2>&1 | grep -E "loaded to slot|uio.*gemv|udmabuf[0-9]:"
(cd ~/kv260gemv && ./bench_gemv2 ~/vectors 2>&1 | grep -E "EXACTNESS|lm_head")
