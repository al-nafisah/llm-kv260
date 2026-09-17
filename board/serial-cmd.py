#!/usr/bin/env python3
"""Run shell commands on a KV260 over its serial console, without touching a cable.

    .venv/bin/python board/serial-cmd.py /dev/ttyUSB1 'hostname' 'ip -4 -br addr'

Logs in if it finds a login prompt (user ubuntu, password from KV260_PASSWORD,
which must be set), runs each command, prints its output, leaves the session
logged in. The console is the one interface that works when the network is
the thing being debugged.
"""

import os
import re
import sys
import termios
import time

import pexpect
import pexpect.fdpexpect

PROMPT = r"\$ $"


def open_console(dev):
    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0                      # iflag: raw
    attrs[1] = 0                      # oflag
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL   # cflag
    attrs[3] = 0                      # lflag: no echo, no canonical
    attrs[4] = attrs[5] = termios.B115200
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    os.set_blocking(fd, True)
    return fd


def main():
    dev, cmds = sys.argv[1], sys.argv[2:]
    password = os.environ.get("KV260_PASSWORD")
    if not password:
        sys.exit("set KV260_PASSWORD")
    fd = open_console(dev)
    con = pexpect.fdpexpect.fdspawn(fd, timeout=10, encoding="utf-8", codec_errors="replace")
    con.send("\r")
    time.sleep(0.5)
    con.send("\r")
    state = con.expect([r"login: $", r"[Pp]assword: $", PROMPT, pexpect.TIMEOUT], timeout=6)
    if state == 3:
        print(f"{dev}: nothing answered on the console", file=sys.stderr)
        sys.exit(2)
    if state == 0:
        con.sendline("ubuntu")
        con.expect(r"[Pp]assword: $")
        state = 1
    if state == 1:
        con.sendline(password)
        got = con.expect([PROMPT, r"Login incorrect", r"Current password: $"], timeout=15)
        if got == 1:
            print(f"{dev}: login incorrect (KV260_PASSWORD?)", file=sys.stderr)
            sys.exit(3)
        if got == 2:
            print(f"{dev}: this board still wants its first-login password change", file=sys.stderr)
            sys.exit(4)
    # Quiet prompt so the output is parseable: no colour, known PS1.
    con.sendline("export PS1='\\$ '; stty -echo")
    con.expect(PROMPT)
    for cmd in cmds:
        con.sendline(cmd)
        con.expect(PROMPT, timeout=60)
        out = con.before
        out = re.sub(r"\r", "", out)
        print(f"$ {cmd}\n{out.strip()}")
    con.sendline("stty echo")
    con.expect(PROMPT)


if __name__ == "__main__":
    main()
