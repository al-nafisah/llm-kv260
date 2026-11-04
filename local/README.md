# Local console

A browser page for the four boards, LAN-only: the chat, four board cards fed by
`/stats` once a second (temperature, power, RAM, memory bandwidth, PL use) and a
pipeline card with the last chat's per-token split. It is the retired public
page pointed at board 1 directly; nothing publishes it.

    bash local/serve.sh                                 # http://127.0.0.1:8090
    BOARD=http://10.42.0.218:8080 bash local/serve.sh   # another address (rewrites config.js)

The page reads the API from `config.js` (`?api=http://...` overrides it); the
server answers CORS `*`, so a page on 127.0.0.1 can call the board directly.
Markdown and math come from three pinned cdnjs libraries (marked, DOMPurify,
KaTeX, SHA-384 checked), so the page needs internet for those; without them
replies show as plain text. `board/tp/chat.py` is the terminal equivalent.
