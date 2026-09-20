# The public test page

`index.html` + `config.js` are deployed to the public repo
`AbdullahAlNafisah/llm-kv260-demo` (GitHub Pages:
https://abdullahalnafisah.github.io/llm-kv260-demo/). This repo is private and
GitHub Pages on a private repo needs a paid plan, so the page lives in its own
public repo and nothing else does. `board/tp/publish-demo.sh` pushes it.

The page calls the boards at **https://kv260.al-nafisah.com**: a Cloudflare
named tunnel (`kv260`, config `~/.cloudflared/kv260.yml`) that runs on the PC
as the user service `cloudflared-kv260` and forwards to board 1's port 8080.
GitHub can host the page but cannot reach the LAN, so some route like this is
unavoidable; a named tunnel keeps a stable hostname, where the anonymous
quick tunnels (`cloudflared tunnel --url ...`) registered but never routed
from this PC, twice.

```bash
systemctl --user status cloudflared-kv260      # the route
bash board/tp/serve-tp.sh 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114   # the API
```

The server admits one prompt at a time in arrival order (`/health` shows
`busy` and `queued`; more than 8 waiting gets HTTP 429), so a visitor always
gets the full single-stream rate, which is the point of the four boards.
Cloudflare's managed rules refuse the default `Python-urllib` user agent
(403); curl, the OpenAI SDKs and anything with a real User-Agent pass.
