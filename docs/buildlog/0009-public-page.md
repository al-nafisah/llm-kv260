# 0009 — A public page and a public API: 20.8–21.9 tokens/s from outside

**Status: Attested.** 2026-09-14, measured from the PC through the public
URL. Raw log: `evidence/0009-public-api.txt`.

- Page: https://al-nafisah.github.io/llm-kv260-demo/ (GitHub Pages,
  public repo `llm-kv260-demo`; source in `web/`).
- API: https://kv260.al-nafisah.com — a Cloudflare named tunnel from the PC
  to board 1, OpenAI-compatible, one prompt at a time in arrival order.

Through the public URL, streaming, three repeats of a 22-token prompt with
67 tokens generated: **20.83 / 21.85 / 20.83 t/s** client-side, equal to the
server's own figure — the tunnel adds nothing per token. First token 1.4–2.5 s
(prefill ~0.9 s plus the tunnel's connection setup). CORS preflight from the
Pages origin: 204 with the right headers. Two concurrent requests: the second
waited for the first (`/health` showed `busy`), then ran at full rate.

GitHub Pages is static hosting and the boards are on a LAN, so the page needs
a route in. The anonymous quick tunnel registered and never routed, twice,
from this PC; the named tunnel holds a stable hostname. Open WebUI is a fine
front end, but it makes its own extra model calls per message (titles, tags,
follow-ups), which on a one-stream server queue behind the user's own
request, and a static page keeps the pipeline for the tester. It can be added
later behind the same API if a fuller UI is wanted.
