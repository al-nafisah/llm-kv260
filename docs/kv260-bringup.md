# Bringing up a KV260 for this project

From a blank microSD to a board that runs its quarter of the model. Done
four times (2026-09-12 to 14); everything below is what actually worked.

Shell note: the PC runs fish; `bash` blocks are for the boards. The Claude
Code tool shell is zsh, whose 1-indexed arrays and unsplit variables have
both broken launchers here — run the `board/tp/*.sh` scripts with `bash`.

## A. Card and first boot

Image: Ubuntu 22.04 for Kria, `iot-limerick-kria-classic-desktop-2204-20240304-165.img.xz`.

```fish
lsblk -o NAME,SIZE,TRAN,RM,MOUNTPOINTS            # find the card; never a SATA disk
set IMG ~/Downloads/iot-limerick-kria-classic-desktop-2204-20240304-165.img.xz
xzcat $IMG | sudo dd of=/dev/sdX bs=4M iflag=fullblock status=progress conv=fsync
xzcat $IMG | sha256sum; sudo head -c (xz --robot --list $IMG | awk '/^totals/{print $5}') /dev/sdX | sha256sum   # must match
```

Boot with Ethernet on the switch (the PC shares its `eno1`, 10.42.0.1/24, and
hands out leases) and the micro-USB console connected. The board appears as
hostname `kria`:

```fish
journalctl -u NetworkManager --since "10 min ago" | grep DHCPACK | grep kria
```

## B. First login, hands-free

The image is `ubuntu`/`ubuntu` with a forced password change. `board/first-login.py`
drives it over ssh: new password, this PC's key, hostname, no desktop.

```fish
.venv/bin/python board/first-login.py 10.42.0.218 kv260-03
ssh ubuntu@10.42.0.218 'echo <password> | sudo -S reboot'
```

Then pin time sync to IPv4 — the board has link-local IPv6 only, the NTP name
resolves to IPv6 first, and without this TLS later fails with "certificate is
not yet valid":

```bash
sudo sh -c 'printf "[Time]\nNTP=185.125.190.56 185.125.190.57 185.125.190.58\n" > /etc/systemd/timesyncd.conf'
sudo systemctl restart systemd-timesyncd
```

## C. The engine, driver, runtime

Ship a kit and run one script; it ends with the engine's exactness check.

```fish
# kit.tgz = board/ + hw/kv260_gemv2/out/gemv2 + hls/int4_core/vectors + the u-dma-buf source (github.com/ikwzm/udmabuf)
scp kit.tgz board/board-setup.sh ~/llama-3.2-1b-instruct-q4_0-pure.gguf ubuntu@10.42.0.218:~/
ssh ubuntu@10.42.0.218 'bash ~/board-setup.sh'                 # ... EXACTNESS: PASS
# a board whose sudo asks for a password: SUDO_PASS=<password> bash ~/board-setup.sh

# libllama for the runtime (tokenizer, GGUF, chat template), ~20 min, detached
scp cmake-4.4.3-linux-aarch64.tar.gz llamacpp-src.tar.gz board/board-llama-build.sh ubuntu@10.42.0.218:~/
ssh ubuntu@10.42.0.218 'setsid nohup bash ~/board-llama-build.sh >/dev/null 2>&1 < /dev/null &'
ssh ubuntu@10.42.0.218 'cd ~/board/tp && make'
```

(cmake is not on the image and apt on it needs `-o Acquire::ForceIPv4=true`;
the static cmake tarball avoids both.)

## D. After a board reboot

Overlay, memory pools and the AXI-width fix are all lost:

```bash
sudo bash ~/board/load-overlay.sh ~/gemv2          # the app name is the .bit.bin's
```

then from the PC `bash board/tp/serve-tp.sh <four addresses>`. The runtime
detects which engine the overlay provides (`gemv_mac` or `gemv_q4_0` in
`/sys/class/uio/*/name`) and lays the weights out for it, so a cluster can
be mixed while overlays are being changed; it runs at the slower engine's
speed. Only the gemv2 overlay is built from this repo now.
`~/kv260gemv/bench_gemv2 ~/vectors` proves gemv2 (needs the n32 vector cases
in `~/vectors`).

## Traps, each of which cost time once

- **A board that DHCPs but never answers ARP has a bad link, not a bad
  config.** Board 1 did that on one switch port (RX errors, `Link detected:
  no`); a different port fixed it. `board/serial-cmd.py /dev/ttyUSBn 'cmd'`
  runs commands over the console so this can be seen without moving cables.
- **The overlay must carry the `xlnx,afi-fpga` node** (`hw/kv260_gemv2/pl.dtsi`).
  Without it the PS AXI master stays at 128 bits, a 32-bit control port
  loses most register writes, and the engine "finishes" in 2 µs.
- **One CMA allocation above ~512 MiB fails** with 989 MiB free; three pools
  (512 + 256 + 128) succeed.
- **`llama-quantize` keeps `output.weight` at Q6_K** unless told otherwise;
  the engine only takes Q4_0: `--output-tensor-type q4_0 --token-embedding-type q4_0`.
- **`ssh` in the background without `-n`** stalls and the remote process
  blocks on a full stderr pipe; a detached remote process needs
  `setsid nohup … < /dev/null &`.
- **A bench on a board where the runtime is resident must use `udmabuf2`.**
  `udmabuf0` holds the loaded shard and `bench_gemv2` bump-allocates from the
  pool's start, so it would write random bytes over it. It uses `udmabuf0` by
  default: stop the runtime first, or pass `udmabuf2` (it fits everything but
  `lm_head`'s 37 MB).
- **gemv2's FIFOs outlive a process.** A process that dies with a fetch in
  flight leaves its bytes on chip; the runtime and the bench drain them at
  start, and the drain must read all eight FIFOs at once (buildlog 0012).
  If a gemv2 process ever reports the MAC kernel busy at start, reload the
  overlay.
- **A busy board gives a wrong number, not a slow one**: a benchmark read
  21 t/s with `unattended-upgrades` running and 305 t/s idle. Check idle
  before believing a number.
  Ubuntu's apt-daily timers, `unattended-upgrades` and the login MOTD's
  `apt-check` ran for hours on three boards (no swap, four A53s) and, since
  every all-reduce waits for the slowest board, took the four-board stream
  from 22 t/s to 8. `sudo bash ~/board/disable-auto-updates.sh` turns all of
  them off for good; apt still works by hand.
- **A firewall change on the PC takes all four boards down at once, and says
  nothing.** Enabling UFW for an unrelated reason (an RDP-over-Tailscale
  rule) left no allow rule for `eno1`, and the PC is the boards' DHCP
  server: every lease request was dropped and all four boards lost their
  addresses together. Nothing in the log named it —
  `/etc/ufw/after.rules` sends DHCP to the default policy through
  `ufw-skip-to-policy-input`, which bypasses the log by design, so 1851
  logged blocks contained not one of the packets that mattered. ICMP kept
  working (allowed in `before.rules`), so the boards looked electrically
  fine, and `ip -br link` on the console showed `eth0` UP with LOWER_UP and
  no address. `sudo ufw allow in on eno1` fixes it. **Four boards failing
  simultaneously is a symptom of this PC, not of four boards**: a
  board-side fault does not synchronise.
- **Recabling the switch reshuffles the DHCP leases.** After the boards were
  unplugged and returned to different ports, kv260-02 and kv260-03 swapped
  addresses. Rank order in `serve-tp.sh` is argument order, so the old
  command still runs — it just gives the ranks to different boards, and
  `/stats` hostnames stop matching the rank numbers. The MAC is the
  identity and the address is not (`ip neigh | grep 00:0a:35`, and the
  table below). A moved address also invalidates `~/.ssh/known_hosts`,
  which arrives as the `@@@@` host-key banner and looks like a board fault
  rather than a lease change: `ssh-keygen -R <address>` per moved board.

## The four boards

| board | MAC (fixed) | address (2026-09-22) | sudo |
|---|---|---|---|
| kv260-01 | `00:0a:35:2a:d5:e6` | 10.42.0.137 | passwordless |
| kv260-02 | `00:0a:35:2a:d4:f9` | 10.42.0.218 | passwordless |
| kv260-03 | `00:0a:35:2a:d5:41` | 10.42.0.217 | needs a password |
| kv260-04 | `00:0a:35:2a:d5:cf` | 10.42.0.114 | needs a password |

The MAC column is the identity. Addresses are DHCP leases from the PC and
move whenever the boards are recabled or reboot in a different order, and
kv260-02 and kv260-03 have already traded places once. So
`ip neigh | grep 00:0a:35` is the only thing that is always right, and
`sudo` rights follow the hostname, not the address. The serial consoles
enumerate in whatever order the FT4232H adapters are plugged in; read
`hostname` over the console before trusting a `/dev/ttyUSB*` number. All four: ZynqMP KV260 revB, 4× Cortex-A53 at 1333 MHz,
3.8 GiB RAM, no swap, kernel 5.15.0-1027-xilinx-zynqmp.

## Exchange-related board settings (buildlog 0011)

Two things on a stock board slow the four-way exchange. Both scripts are
run on the board as root and persist across reboots:

- `sudo bash ~/board/disable-dp.sh` — unbinds the DisplayPort subsystem,
  which with no monitor attached raises ~10,500 interrupts per second on
  CPU 0. Installs `unbind-dp.service`.
- `sudo bash ~/board/jumbo-frames.sh` — MTU 9000 on eth0, now and at
  boot. It writes the netplan file for the next boot (the MTU line into
  `99-direct-link.yaml` on boards 1 and 2, a `99-jumbo-frames.yaml` with
  `dhcp4: true` on boards that have only cloud-init's file) and changes the
  live MTU from a detached `ip link set eth0 down / mtu 9000 / up`, because
  the macb driver refuses an MTU change while the interface is up. The ssh
  session closes and the board answers again within seconds. It never runs
  `netplan apply`: an earlier version did, and on two boards that stopped
  NetworkManager for good, until the console ran
  `sudo systemctl start NetworkManager`. The `dhcp4` line matters: an eth0
  that netplan defines without it boots with no IPv4 address.

Applied on all four boards since 2026-09-14; every ordered pair passes
9000-byte packets with do-not-fragment set (`buildlog/evidence/0012-board-settings.txt`).
To do both on a board that asks for a sudo password, from the PC:

```bash
ssh -t ubuntu@<board> 'sudo bash ~/board/disable-dp.sh && sudo bash ~/board/jumbo-frames.sh'
```

The runtime pins itself to CPU 2 (`tp_llama --cpu`, default 2; `-1` leaves
placement to the scheduler).
