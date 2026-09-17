#!/usr/bin/env bash
# On the board. Turn off Ubuntu's automatic package work: the apt-daily
# timers, unattended-upgrades, and update-notifier's apt-check that the
# login MOTD runs. On a 4-core board with no swap these take one or two
# cores for hours, and because every all-reduce waits for the slowest
# board, one busy board drags the whole stream from ~22 t/s to ~8.
# Stops whatever is running (only after checking dpkg is not mid-install),
# then makes sure nothing restarts it. apt still works by hand. Idempotent.
#
#   sudo bash ~/board/disable-auto-updates.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

if pgrep -x dpkg >/dev/null; then
    echo "dpkg is installing a package; let it finish and run this again" >&2
    exit 1
fi

# The timers that start apt-daily.service and apt-daily-upgrade.service.
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer >/dev/null 2>&1 || true
systemctl stop apt-daily.service apt-daily-upgrade.service >/dev/null 2>&1 || true
systemctl mask apt-daily.service apt-daily-upgrade.service >/dev/null 2>&1 || true
# unattended-upgrades.service only exists to finish an upgrade at shutdown.
systemctl disable --now unattended-upgrades.service >/dev/null 2>&1 || true

# The periodic settings apt reads (this is what dpkg-reconfigure writes).
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Download-Upgradeable-Packages "0";
APT::Periodic::AutocleanInterval "0";
APT::Periodic::Unattended-Upgrade "0";
EOF

# The MOTD script that runs apt-check at every ssh login when its stamp is stale.
[ -f /etc/update-motd.d/90-updates-available ] && chmod -x /etc/update-motd.d/90-updates-available
[ -f /etc/update-motd.d/91-release-upgrade ] && chmod -x /etc/update-motd.d/91-release-upgrade

# Whatever is still running from before.
pkill -f /usr/lib/apt/apt.systemd.daily >/dev/null 2>&1 || true
pkill -f /usr/bin/unattended-upgrade >/dev/null 2>&1 || true
pkill -f /usr/lib/update-notifier/apt-check >/dev/null 2>&1 || true
sleep 2
pkill -9 -f "apt.systemd.daily|unattended-upgrade|update-notifier/apt-check" >/dev/null 2>&1 || true
# If a stop landed between packages, dpkg's database says so; this settles it.
dpkg --configure -a >/dev/null 2>&1 || true

echo "--- timers: $(systemctl is-enabled apt-daily.timer apt-daily-upgrade.timer 2>&1 | tr '\n' ' ')"
echo "--- periodic: $(tr '\n' ' ' < /etc/apt/apt.conf.d/20auto-upgrades)"
echo "--- still running: $(pgrep -a -f 'apt.systemd.daily|unattended-upgrade|apt-check' || echo none)"
echo "--- load: $(cut -d' ' -f1-3 /proc/loadavg)"
