#!/usr/bin/env bash
# Serve the local console on http://127.0.0.1:8090. The page calls the API
# on board 1 directly (the server answers CORS *), so this only hands out
# the files. BOARD=http://10.42.0.x:8080 points the page at another address
# by rewriting config.js before serving; the default is board 1.
#
#   bash local/serve.sh
#   BOARD=http://10.42.0.218:8080 bash local/serve.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
board=${BOARD:-http://10.42.0.137:8080}
printf 'window.KV260_API = "%s";\n' "$board" > "$here/config.js"
echo "console: http://127.0.0.1:8090  (API $board)"
exec python3 -m http.server --bind 127.0.0.1 8090 --directory "$here"
