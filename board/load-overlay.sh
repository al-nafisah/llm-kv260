#!/usr/bin/env bash
# On the board. Loads the gemv2 overlay and the contiguous-memory pools, then
# proves both are up. Idempotent.
#
#   sudo bash ~/board/load-overlay.sh ~/gemv2         # dir from hw/kv260_gemv2/package.sh
# The app name is the .bit.bin's basename.
#
# Pools: 512 + 256 + 128 MiB. Measured on kv260-01: a single allocation above
# ~512 MiB fails even with 989 MiB of CMA free (the area is fragmented by
# boot-time allocations), while these three succeed. Llama-3.2-1B Q4_0 is
# 728 MiB; the backend splits it across pools.
set -euo pipefail
fw=${1:?firmware dir with <name>.bit.bin, <name>.dtbo, shell.json}
name=$(basename "$(ls "$fw"/*.bit.bin | head -1)" .bit.bin)

if [ "$(id -u)" != 0 ]; then echo "run with sudo" >&2; exit 1; fi

mkdir -p /lib/firmware/xilinx/$name
cp "$fw"/$name.bit.bin "$fw"/$name.dtbo "$fw"/shell.json /lib/firmware/xilinx/$name/

# xmutil owns the PL. Unload whatever is there first (the starter-kit app by default).
xmutil unloadapp >/dev/null 2>&1 || true
xmutil loadapp $name
sleep 1

echo "--- fpga: $(cat /sys/class/fpga_manager/fpga0/state)"
echo "--- pl0 clock:"
grep -E "pl0_ref " /sys/kernel/debug/clk/clk_summary || true
echo "--- uio:"
for u in /sys/class/uio/uio*; do echo "  $(basename $u) $(cat $u/name) $(cat $u/maps/map0/addr 2>/dev/null)"; done | grep -E "gemv|axi-pmon" || true

# Contiguous pools.
if ! lsmod | grep -q u_dma_buf; then
    insmod /home/ubuntu/udmabuf/u-dma-buf.ko udmabuf0=536870912 udmabuf1=268435456 udmabuf2=134217728
fi
for i in 0 1 2; do
    echo "--- udmabuf$i: $(cat /sys/class/u-dma-buf/udmabuf$i/size) B at $(cat /sys/class/u-dma-buf/udmabuf$i/phys_addr)"
done
chmod a+rw /dev/udmabuf* /dev/uio*
