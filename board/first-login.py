#!/usr/bin/env python3
"""First login on a fresh Kria Ubuntu board, over ssh, hands-free.

    .venv/bin/python board/first-login.py 10.42.0.218 kv260-03

The image ships as ubuntu/ubuntu and forces a password change at first
login. This drives that dialogue, installs this PC's ssh key, names the
board, and drops the graphical target (several hundred MB on a 4 GB board).
It does NOT reboot; the caller does, once it has read the output.

New password comes from KV260_PASSWORD, which must be set.
"""

import os
import sys

import pexpect

SSH = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no"


def main():
    ip, hostname = sys.argv[1], sys.argv[2]
    new = os.environ.get("KV260_PASSWORD")
    if not new:
        sys.exit("set KV260_PASSWORD")
    pubkey = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()

    child = pexpect.spawn(f"{SSH} ubuntu@{ip}", timeout=30, encoding="utf-8")
    child.expect(r"[Pp]assword: ")
    child.sendline("ubuntu")
    i = child.expect([r"Current password: ", r"\$ $", r"Permission denied"])
    if i == 2:
        print(f"{ip}: ubuntu/ubuntu refused -- already set up? try KV260_PASSWORD", file=sys.stderr)
        sys.exit(3)
    if i == 0:
        child.sendline("ubuntu")
        child.expect(r"New password: ")
        child.sendline(new)
        child.expect(r"Retype new password: ")
        child.sendline(new)
        # PAM closes the session after a forced change; log in again.
        child.expect(pexpect.EOF)
        child = pexpect.spawn(f"{SSH} ubuntu@{ip}", timeout=30, encoding="utf-8")
        child.expect(r"[Pp]assword: ")
        child.sendline(new)
        child.expect(r"\$ $")
        print(f"{ip}: password changed")
    else:
        print(f"{ip}: no forced change (password was already ubuntu and stays so?)")

    def run(cmd, sudo=False):
        child.sendline(cmd)
        if sudo:
            j = child.expect([r"\[sudo\] password for ubuntu: ", r"\$ $"])
            if j == 0:
                child.sendline(new)
                child.expect(r"\$ $")
        else:
            child.expect(r"\$ $")

    run("mkdir -p ~/.ssh && chmod 700 ~/.ssh")
    run(f"grep -qF '{pubkey}' ~/.ssh/authorized_keys 2>/dev/null || echo '{pubkey}' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys")
    run(f"sudo hostnamectl set-hostname {hostname}", sudo=True)
    run("sudo systemctl set-default multi-user.target", sudo=True)
    run("hostname; systemctl get-default")
    print(child.before.strip().splitlines()[-2:])
    child.sendline("exit")
    child.expect(pexpect.EOF)
    print(f"{ip}: key installed, hostname {hostname}, multi-user target. Reboot to apply.")


if __name__ == "__main__":
    main()
