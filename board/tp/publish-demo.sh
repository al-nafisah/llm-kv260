#!/usr/bin/env bash
# Push web/index.html + web/config.js (with the current tunnel URL) to the public demo repo.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
url=${1:-$(cat ~/.cache/llm-kv260/tunnel-url.txt)}
printf 'window.KV260_API = "%s";\n' "$url" > "$here/web/config.js"
d=$(mktemp -d); git clone -q https://github.com/AbdullahAlNafisah/llm-kv260-demo.git "$d"
cp "$here/web/index.html" "$here/web/config.js" "$d/"; touch "$d/.nojekyll"
git -C "$d" add -A && git -C "$d" -c user.name=Nafisah -c user.email=58102127+AbdullahAlNafisah@users.noreply.github.com commit -q -m "Update page (tunnel $url)" && git -C "$d" push -q
echo "published: https://abdullahalnafisah.github.io/llm-kv260-demo/  (API $url)"
