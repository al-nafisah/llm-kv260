#!/usr/bin/env bash
# On a board, as root: eth0 at MTU 9000, now and at every boot. An 8 KB
# exchange message is then one frame instead of six (buildlog/0011).
#
#   sudo bash ~/board/jumbo-frames.sh
#
# No `netplan apply`: on two boards it stopped NetworkManager and did not
# start it again, and they were off the network until the serial console
# started it (buildlog/0011). The netplan file is written for the next boot
# only, and the live MTU is changed with ip link. The macb driver refuses an
# MTU change while the link is up, so the link goes down for a few seconds,
# from a detached process: the ssh session may stall or drop.
#
# Boards 1 and 2 carry /etc/netplan/99-direct-link.yaml (with their 10.0.0.x
# address), which gets the MTU line. The others have only cloud-init's file,
# which names no interface, and get 99-jumbo-frames.yaml. Its dhcp4 line
# matters: an eth0 that netplan defines without it boots with no IPv4 address.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

direct=/etc/netplan/99-direct-link.yaml
jumbo=/etc/netplan/99-jumbo-frames.yaml
if [ -f $direct ]; then
    grep -q "mtu:" $direct || sed -i 's/^    eth0:$/    eth0:\n      mtu: 9000/' $direct
    grep -q "mtu: 9000" $direct || { echo "could not edit $direct" >&2; exit 1; }
    echo "$(hostname): $direct sets MTU 9000 at boot"
else
    umask 077
    cat > $jumbo <<'YAML'
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: true
      mtu: 9000
YAML
    echo "$(hostname): $jumbo sets MTU 9000 at boot"
fi

if [ "$(cat /sys/class/net/eth0/mtu)" = 9000 ]; then
    echo "$(hostname): eth0 is already at MTU 9000"
    exit 0
fi
setsid nohup sh -c 'sleep 2; ip link set eth0 down; ip link set eth0 mtu 9000; ip link set eth0 up' > /dev/null 2>&1 < /dev/null &
echo "$(hostname): in 2 s eth0 goes down, to MTU 9000, and up; the board answers again within ~10 s"
