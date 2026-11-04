# 0007 — Four boards, one stream: tensor-parallel decode at 22 tokens/s

**Status: Attested.** 2026-09-13/14, `kv260-01..04` on one gigabit switch,
each with the phase 4 engine (`0006`). Raw logs: `evidence/0007-*`. Runtime:
`board/tp/tp_llama.cpp`; design in [../fabric-engine.md](../fabric-engine.md#four-boards-one-stream).

## Result

Llama-3.2-1B all-Q4_0, greedy, prompt "The three primary colors are", 64
tokens, three repeats per width (`evidence/0007-tp-scaling.txt`). The text
is identical at every width and matches the single-board llama.cpp output.

| boards | tokens/s | per token: engine | exchange | CPU |
|---|---|---|---|---|
| 1 | 10.49, 10.49, 10.49 | 65.3 ms | 0 | 30.0 ms |
| 2 | 17.37, 17.45, 17.39 | 33.7 ms | 7.2–7.5 ms | 16.4 ms |
| 4 | **21.36, 23.84, 21.61** | 17.8–18.2 ms | 15.3–20.3 ms | 8.7 ms |

For reference, llama.cpp on one board with the same engine: 9.03 t/s
(`0006`). Two things happened between 9.0 and 22:

- **A runtime with no scheduler.** `tp_llama` at one rank is already
  10.49 t/s: the same engine, the same 65 ms of weight streaming, but the
  ops between matmuls are plain loops on one core instead of 86 CPU-backend
  starts per token. Fusing q/k/v and gate/up into one engine call each also
  cut 113 calls to 65.
- **Tensor parallelism.** Each rank streams a quarter of every matrix; the
  engine time divides almost exactly (65 → 33.7 → 17.8 ms), attention
  divides with the heads, and the price is 33 exchanges of 8 KB per token
  over the switch.

At four boards the exchange is now the largest term. The 0.31 ms
round-trip measured in `0005` predicts ~10 ms for 33 exchanges; measured
is 15–20 ms and it jitters, so the all-to-all over TCP is running at about
half its potential. That, not the fabric, is the next thing to optimise.

## What the four boards do per token

Rank *r* holds rows `[rN/4, (r+1)N/4)` of q, k, v, gate, up and lm_head
(contiguous byte ranges of the Q4_0 file, copied into engine memory at
load) and columns `[rK/4, (r+1)K/4)` of attn_output and ffn_down (18-byte
blocks sliced out of each row). Per layer: RMSNorm (replicated), one engine
pass for q|k|v, RoPE, attention over its 8 query heads and 2 KV heads (no
traffic: the heads a rank owns are exactly the ones its KV shard serves),
one engine pass for the output-projection partial, **all-reduce**,
residual, RMSNorm, one pass for gate|up, SiLU, one pass for the down
partial, **all-reduce**, residual. Then the lm_head shard, a local argmax,
and one 8-byte exchange to pick the global one. Partials are summed in
rank order on every rank, so the replicated hidden state is bit-identical
everywhere, which is also why the four ranks print the same text.

Contiguous memory per rank: 347.6 MB at two ranks, ~174 MB at four; the
whole 663 MiB model at one rank still fits the three pools.

## What went wrong on the way

1. **Cold-start ordering.** Ranks that had the model in page cache loaded
   in 0.3 s; the others took 10–14 s. Every rank must tolerate peers that
   are seconds late.
2. **The UDP exchange deadlocked, twice, differently.** First: a rank one
   step behind asked for a resend of the previous sequence and the others
   only kept the current one. Second, after fixing that: a starved rank
   never sent its resend request because the request was triggered by a
   quiet socket, and three peers' resend rounds kept its socket busy.
   After fixing that too, the four-way run *completed* — at 1.26 t/s, with
   767 ms of every token spent recovering lost packets, none of which the
   NICs counted as drops. Replaced by a persistent TCP mesh (rank *r*
   accepts from higher ranks, connects to lower ones; `TCP_NODELAY`),
   which has no loss path: 22 t/s on the first run.
3. **`ssh` in the background without `-n`** contends for the launcher's
   stdin, gets stopped, stops relaying, and the remote rank blocks on a
   full stderr pipe — every "hung" run had exactly 25,469 bytes of log.
   `board/tp/run-tp.sh` uses `ssh -n`.
4. **Two launcher bugs of my own** in the tool shell (zsh): 1-indexed
   arrays sent rank 1 its own address, and unsplit `$ips` launched the
   multi-board runs as single-rank. Both showed up as self-delivered
   packets and empty logs; neither touched the boards.

## Next

1. **The exchange**: 33 × ~0.5 ms → the link's 0.3 ms. Halve the payload
   (float16 partials), reduce the count (a fused residual path needs one
   all-reduce per layer with a different sharding), and busy-poll instead
   of blocking reads. Worth ~30 t/s.
2. **Bandwidth per board** (`0006` next-steps) still applies: 11.6 → 16
   GB/s takes the engine term from 18 to 13 ms.
3. **Bigger models.** At four boards the per-rank shard of a 3B model is
   ~430 MiB and fits; this is the first configuration on which Llama-3.2-3B
   would run at a usable rate.

## Corrections (2026-11-04)

- The per-token columns of the table are rank 0's. The other ranks in the
  same runs: two boards engine 34.7–34.9 ms, exchange 6.1–6.2, CPU 16.5;
  four boards engine 17.8–20.9, exchange 13.9–20.4, CPU 8.6–11.3
  (`evidence/0007-tp-scaling.txt`).
- "The text is identical at every width": the evidence holds about the
  first 120 characters of rank 0's text at each width (lines 9, 19, 35), and
  those are identical; the full texts were not saved, and no llama.cpp
  output is logged to compare with. Likewise "the four ranks print the same
  text": only rank 0's text is in the log; that every rank holds the same
  hidden state follows from the rank-order sums and is not a logged
  comparison.
- "33 exchanges of 8 KB per token" is 32 all-reduces of an 8 KB vector and
  one 8-byte argmax exchange. Since `0008` that exchange carries each
  rank's top 40 candidates and its board stats, 356 bytes per rank
  (`board/tp/tp_llama.cpp:899`); the 0007 build itself is not in the tree.
- `0005`, cited for the 0.31 ms round trip, is not part of this
  repository; that figure has no evidence file here.
