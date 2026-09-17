#!/usr/bin/env bash
# On a board, as root: unbind the DisplayPort subsystem and keep it unbound
# across reboots. With no monitor attached it raises ~10,500 interrupts per
# second on CPU 0, all day (buildlog/0011). Re-enable with
#   systemctl disable --now unbind-dp && echo fd4a0000.display > /sys/bus/platform/drivers/zynqmp-display/bind
set -e
cat > /etc/systemd/system/unbind-dp.service <<'UNIT'
[Unit]
Description=Unbind the ZynqMP DisplayPort subsystem (interrupt storm without a monitor)

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'echo fd4a0000.display > /sys/bus/platform/drivers/zynqmp-display/unbind || true'

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now unbind-dp.service
sleep 1
a=$(awk '/fd4a0000/ {print $2+$3+$4+$5}' /proc/interrupts); sleep 2; b=$(awk '/fd4a0000/ {print $2+$3+$4+$5}' /proc/interrupts)
echo "$(hostname): display interrupts in 2 s: $((b - a))"
