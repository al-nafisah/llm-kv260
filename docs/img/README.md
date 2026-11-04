# Figures

One figure set for the README. Each
figure is an SVG source with a PDF (vector, fonts embedded, for LaTeX) and a
2x PNG (1600 px wide) beside it. All four use a white background, 800 px
wide at 1x, Arial or a metric-compatible face (Liberation Sans), and one
palette for three roles: **blue** is the FPGA engine, **orange** the network
exchange, **aqua** the CPU (one A53 core).

| figure | shows | source |
|---|---|---|
| `system` | four boards and the PC on one gigabit switch; what each board runs; how a matrix is split by output rows or by input columns; one token's sequence of engine calls, attention and the 33 exchanges | hand-written SVG; facts from `board/tp/tp_llama.cpp` (sharding, exchange, forward) and buildlog 0013 (173.8 MB per board) |
| `engine` | gemv2 on one board: DDR4, four 128-bit ports, the fetch kernel's four readers, eight 256 KiB UltraRAM FIFOs, eight MAC units, activations in and results out on one shared port, the runtime naming the next matrix (prefetch), and the per-block arithmetic | hand-written SVG; facts from `hls/gemv2/`, `hls/int4_core/gemv_core.h`, `hw/kv260_gemv2/build.tcl`, `board/tp/tp_llama.cpp` |
| `token` | engine, exchange and CPU milliseconds per generated token on board 1 (rank 0), every run on 1, 2 and 4 boards, with each run's tokens/s | `make_figs.py` from `docs/buildlog/evidence/0013-tp-runs.txt` |
| `scaling` | tokens/s and joules per token for every run on 1, 2 and 4 boards, plus the same board without its engine (llama.cpp `llama-bench`, CPU only) | `make_figs.py` from `0013-tp-runs.txt`, `0013-phases.txt`, `0013-power-kv260-0N.txt`, `0013-cpu-llama-bench.txt`, checked against `0013-session-report.txt` |

`demo.gif` is not built by `make_figs.py`.

## Regenerate

```bash
python3 docs/img/make_figs.py
```

It parses the evidence files, writes `token.svg` and `scaling.svg`, and then
converts all four SVGs to `.pdf` and 2x `.png` with `rsvg-convert` (librsvg).
It needs only the Python standard library and `rsvg-convert`; it prints every
value it plotted. No value is typed into the script: tokens/s and the time
split are rank 0's summary lines, power is recomputed from the raw INA260
samples, and the run counts, prompt length, token count and `llama-bench`
flags that the captions state are asserted against the logs. Power and
energy per token must agree with `0013-session-report.txt` to its printed
precision or the script stops. Output is byte-for-byte reproducible (the PDF
date is pinned through `SOURCE_DATE_EPOCH`).

The two diagrams are edited by hand; rerun the script afterwards to refresh
their PDF and PNG.

## Reading the charts

- **token.** The three terms are `run-tp.sh`'s per-token means on rank 0:
  each bucket's total over the run divided by the 400 generated tokens. The
  buckets also count the prompt's forwards and leave out a little untimed
  work (sampling, text output), so a bar is close to, not equal to, the wall
  time per token; each run's measured tokens/s is printed beside its bar.
  Engine time is the wall time of each engine call, including quantizing
  its input and the cache maintenance on the CPU. On one board the exchange
  is 0.2 ms, too thin to draw, so it is named under the group label.
- **scaling.** Dots are individual runs, offset sideways so the three
  stay visible; there are no error bars. Module power is the mean of each
  running K26 module's INA260 samples (about 10 per second) inside the
  decode window that `board/tp/bench/session-report.py` uses, summed over
  the boards that ran; joules per token is that power divided by the run's
  tokens/s. Carrier cards, fans, power supplies and the switch are not
  measured. The CPU-only point is `llama-bench` (tg400 from an empty
  context, four A53 threads, `-dev none`), which reports a mean and a
  standard deviation of three runs, not the runs; it is different software
  from the engine path and describes this board without its fabric, not
  llama.cpp in general.

## Checks done when the set was made (2026-11-04)

- Each SVG rendered with headless Chrome (`google-chrome-stable --headless=new
  --force-device-scale-factor=2 --window-size=800,<height> --screenshot=…
  file://…/<name>.svg`) and with `rsvg-convert`, and the images inspected
  for clipped or colliding text.
- `pdfinfo` and `pdffonts` on each PDF: one page, Liberation Sans embedded.
- Every number in the four figures checked against the tables of
  buildlog 0013 (`docs/buildlog/0013-release-session.md`).
- The categorical palette (`#2a78d6`, `#eb6834`, `#1baf7a`) passes the
  colour-vision checks on white; aqua is under 3:1 against white, so every
  aqua mark carries a direct label.
